"""Deterministic evidence-level scoring, independent of an LLM judge."""
from math import log2
from statistics import mean


def score_case(case: dict, results: list[dict], k: int) -> dict:
    groups = [set(group) for group in case['required']]
    selected = results[:k]
    if not groups:
        return {'recall': None, 'precision': None, 'mrr': None,
                'complete': None, 'false_evidence_rate': int(bool(selected)), 'retrieved': len(selected)}
    found, first, relevant, coverage, complete_ids = set(), 0, 0, {}, set()
    for rank, chunk in enumerate(selected, 1):
        labels=set(chunk.get('evidence_ids',[]))
        complete_ids |= labels
        overlaps=set(labels)
        for span in chunk.get('evidence_spans',[]):
            eid=span['id']
            overlaps.add(eid)
            coverage.setdefault(eid,[]).append((span['covered_start'],span['covered_end']))
            cursor=span['start']
            for start,end in sorted(coverage[eid]):
                if start>cursor:
                    break
                cursor=max(cursor,end)
            if cursor>=span['end']:
                complete_ids.add(eid)
        matched={i for i,group in enumerate(groups) if group & complete_ids}
        relevant += any(group & overlaps for group in groups)
        if matched and not first:
            first = rank
        found |= matched
    return {'recall': len(found)/len(groups), 'precision': relevant/len(selected) if selected else 0.,
            'mrr': 1/first if first else 0.,
            'complete': len(found) == len(groups), 'false_evidence_rate': None, 'retrieved': len(selected)}


def aggregate(scores: list[dict]) -> dict:
    result = {'cases': len(scores), 'answerable_cases': sum(s['recall'] is not None for s in scores)}
    for key in ['recall','precision','mrr','complete','false_evidence_rate']:
        values = [s[key] for s in scores if s[key] is not None]
        result[key] = mean(values) if values else None
    return result


def case_regressions(baseline: dict,candidate: dict) -> list[dict]:
    candidates={row['id']:row for row in candidate['cases']}
    regressions=[]
    for before in baseline['cases']:
        after=candidates.get(before['id'])
        worsened=[]
        if after is None:
            worsened=['missing_case']
        else:
            for metric in ['recall','precision','mrr','complete']:
                a,b=before['scores'].get(metric),after['scores'].get(metric)
                if a is not None and (b is None or b<a-1e-10):
                    worsened.append(metric)
        if worsened:
            regressions.append({'id':before['id'],'question':before.get('question'),
                                'baseline':before['scores'],'candidate':after['scores'] if after else None,'worsened':worsened})
    return regressions
