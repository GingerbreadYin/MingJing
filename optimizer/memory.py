"""Bounded, attributable experiment memory with an optional Redis cache."""
import hashlib
import json
import os


CONTEXT_CHARS = 1000
NOTE_EXCERPT_CHARS = 160
CACHE_TTL_SECONDS = 3600


def encoded_context(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def compress(scope, rows):
    total = rows[0]['total'] if rows else 0
    context = {'series_id': scope, 'entries': [], 'omitted': total}
    for row in rows:
        payload = json.loads(row['payload'])
        source = {'task_id': row['task_id'], 'seq': row['seq']}
        if row['kind'] == 'series_memory_note':
            text = ' '.join(payload['text'].split())
            entry = {'kind': 'operator_note', 'source': source, 'text': text[:NOTE_EXCERPT_CHARS],
                     'truncated': len(text) > NOTE_EXCERPT_CHARS}
        else:
            summary = payload['round_summary']
            entry = {'kind': 'development_result', 'source': source,
                     'round': summary.get('round'), 'recall': summary['metrics'].get('recall'),
                     'checks': summary['checks'],
                     'case_ids': [case['id'] for case in summary['case_deltas'][:3]]}
        candidate = {'series_id': scope, 'entries': context['entries'] + [entry],
                     'omitted': context['omitted'] - 1}
        if len(encoded_context(candidate)) <= CONTEXT_CHARS:
            context = candidate
    return context


class SessionMemory:
    def __init__(self, store):
        self.store = store
        self.redis = None
        self.redis_error = ()
        url = os.environ.get('RAGOPT_REDIS_URL')
        if url:
            from redis import Redis
            from redis.backoff import NoBackoff
            from redis.exceptions import RedisError
            from redis.retry import Retry

            self.redis_error = (RedisError,)
            self.redis = Redis.from_url(url, decode_responses=True, socket_connect_timeout=.5,
                                        socket_timeout=.5, retry=Retry(NoBackoff(), 0))

    def close(self):
        if self.redis is not None:
            self.redis.close()

    def view(self, task_id, *, for_model=False):
        scope, rows = self.store.memory_sources(task_id, for_model=for_model)
        cache = 'sqlite'
        if self.redis is not None and rows:
            revision = [scope, task_id, for_model, rows[0]['total'], [dict(row) for row in rows]]
            digest = hashlib.sha256(encoded_context(revision).encode()).hexdigest()
            key = 'ragopt:memory:v1:' + digest
            try:
                cached = self.redis.get(key)
                if cached is not None:
                    context = json.loads(cached)
                    if len(cached) <= CONTEXT_CHARS and context.get('series_id') == scope:
                        return {'context': context, 'cache': 'redis_hit'}
                context = compress(scope, rows)
                self.redis.setex(key, CACHE_TTL_SECONDS, encoded_context(context))
                return {'context': context, 'cache': 'redis_rebuilt'}
            except (*self.redis_error, ValueError, AttributeError):
                cache = 'sqlite_fallback'
        return {'context': compress(scope, rows), 'cache': cache}
