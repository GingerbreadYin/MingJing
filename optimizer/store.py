"""SQLite task state; compare-and-set transitions fence approval and workers."""
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import hashlib, json, sqlite3, uuid
from pathlib import Path


class Conflict(ValueError):
    pass


MAX_ROUNDS = 4
MAX_MODEL_CALLS = 60


def now():
    return datetime.now(timezone.utc).isoformat()


def plan_fingerprint(content):
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_plan_integrity(task):
    if 'experiment_plan' not in task and 'plan_revision' not in task:
        return  # v1.0 rows did not persist their revision alongside the plan.
    content = {'plan': task['plan'], 'snapshot': task['snapshot'], 'reason': task['reason'],
               'revision': task.get('plan_revision'), 'experiment_plan': task.get('experiment_plan')}
    if plan_fingerprint(content) != task['plan_hash']:
        raise Conflict('实验计划与已记录版本不一致')


class Store:
    def __init__(self, path: Path):
        self.path=path
        path.parent.mkdir(parents=True,exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_task_kind_seq ON events(task_id, kind, seq);
                CREATE TABLE IF NOT EXISTS budget(id TEXT PRIMARY KEY, task_id TEXT, reserved REAL NOT NULL, actual REAL, status TEXT NOT NULL);
            ''')

    def connect(self):
        db=sqlite3.connect(self.path,timeout=15)
        db.row_factory=sqlite3.Row
        return db

    def acquire_owner(self):
        # Separate lock DB so the execution owner never blocks task transactions.
        db = sqlite3.connect(self.path.with_suffix('.owner.db'), timeout=0, check_same_thread=False)
        try:
            db.execute('BEGIN IMMEDIATE')
        except sqlite3.Error as exc:
            db.close()
            if getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_BUSY:
                raise Conflict('已有服务占用此任务库，不能同时恢复任务') from None
            raise
        return db

    @contextmanager
    def transaction(self):
        db=self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _read(self, db, task_id):
        row=db.execute('SELECT payload FROM tasks WHERE id=?',(task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return json.loads(row['payload'])

    def _save(self, db, task, kind, detail=None):
        task['updated_at']=now()
        db.execute('UPDATE tasks SET payload=? WHERE id=?',(json.dumps(task,ensure_ascii=False),task['id']))
        db.execute('INSERT INTO events(task_id,at,kind,payload) VALUES(?,?,?,?)',
                   (task['id'],now(),kind,json.dumps(detail or {},ensure_ascii=False)))
        return task

    def create(self, goal: str, snapshot: str, *, iterative=False):
        task={'id':uuid.uuid4().hex,'goal':goal,'snapshot':snapshot,'status':'created','created_at':now(),
              'plan':None,'plan_hash':None,'diagnosis':None,'baseline':None,'candidate':None,'error':None}
        if iterative:
            task.update(series_id=task['id'], round_number=1, attempted_configs=[])
        with self.transaction() as db:
            db.execute('INSERT INTO tasks VALUES(?,?)',(task['id'],json.dumps(task,ensure_ascii=False)))
            return self._save(db,task,'created')

    def get(self, task_id):
        with closing(self.connect()) as db:
            return self._read(db,task_id)

    def list(self):
        with closing(self.connect()) as db:
            return [json.loads(r['payload']) for r in db.execute('SELECT payload FROM tasks ORDER BY rowid DESC')]

    def events(self, task_id):
        return self.read_trace(task_id)[1]

    def events_after(self, task_id, cursor, limit=200):
        with closing(self.connect()) as db:
            rows = db.execute('SELECT seq,kind FROM events WHERE task_id=? AND seq>? ORDER BY seq LIMIT ?',
                              (task_id, cursor, limit))
            return [dict(row) for row in rows]

    def read_trace(self, task_id):
        with closing(self.connect()) as db:
            db.execute('BEGIN')
            task = self._read(db, task_id)
            events = [{**dict(r),'payload':json.loads(r['payload'])}
                      for r in db.execute('SELECT * FROM events WHERE task_id=? ORDER BY seq',(task_id,))]
            return task, events

    def event(self, task_id, kind, detail):
        with self.transaction() as db:
            self._read(db,task_id)
            db.execute('INSERT INTO events(task_id,at,kind,payload) VALUES(?,?,?,?)',(task_id,now(),kind,json.dumps(detail,ensure_ascii=False)))

    def memory_sources(self, task_id, *, for_model=False):
        with closing(self.connect()) as db:
            db.execute('BEGIN')
            task = self._read(db, task_id)
            lineage = []
            current = task
            while current and len(lineage) < MAX_ROUNDS:
                lineage.append(current['id'])
                parent_id = current.get('previous_task_id')
                current = self._read(db, parent_id) if parent_id and task.get('series_id') else None
                if current and current.get('series_id') != task['series_id']:
                    break
            placeholders = ','.join('?' for _ in lineage)
            rows = db.execute(f'''
                SELECT seq, task_id, kind, payload, COUNT(*) OVER () AS total FROM events
                WHERE task_id IN ({placeholders}) AND (
                    (kind='series_memory_note' AND (?=0 OR json_extract(payload,'$.include_in_model')=1))
                    OR (kind='awaiting_round_decision' AND json_type(payload,'$.round_summary')='object')
                )
                ORDER BY (kind='awaiting_round_decision') DESC, seq DESC LIMIT 12
            ''', (*lineage, int(for_model))).fetchall()
        return task.get('series_id') or task_id, rows

    def transition(self, task_id, expected, status, **updates):
        with self.transaction() as db:
            task=self._read(db,task_id)
            if task['status'] not in expected:
                raise Conflict(f"当前状态 {task['status']} 不允许此操作")
            task.update(updates,status=status)
            return self._save(db,task,status,updates)

    def propose(self, task_id, plan, reason, pending=False, experiment_plan=None):
        with self.transaction() as db:
            task=self._read(db,task_id)
            if task['status'] not in {'created','diagnosing','proposal_ready','awaiting_approval'}:
                raise Conflict('当前任务不能提交方案')
            content={'plan':plan,'snapshot':task['snapshot'],'reason':reason,'revision':uuid.uuid4().hex}
            if experiment_plan is not None:
                content['experiment_plan'] = experiment_plan
                task.update(experiment_plan=experiment_plan, plan_revision=content['revision'])
            elif 'experiment_plan' in task:
                raise Conflict('结构化实验计划的后续版本也需要完整实验字段')
            fingerprint=plan_fingerprint(content)
            task.update(plan=plan,plan_hash=fingerprint,reason=reason,status='proposal_ready' if pending else 'awaiting_approval')
            return self._save(db,task,'plan_proposed',content)

    def approve(self, task_id, plan_hash, *, resumable=False):
        with self.transaction() as db:
            task=self._read(db,task_id)
            if task['status']!='awaiting_approval' or task['plan_hash']!=plan_hash:
                raise Conflict('审批已过期或任务状态不允许')
            check_plan_integrity(task)
            task.update(status='approved',approved_hash=plan_hash)
            if resumable:
                task['execution_protocol'] = 1
            return self._save(db,task,'approved',{'plan_hash':plan_hash})

    def claim_execution(self, task_id):
        with self.transaction() as db:
            task = self._read(db, task_id)
            if task['status'] != 'approved' or task.get('approved_hash') != task['plan_hash']:
                raise Conflict('当前方案未获批准或审批已过期')
            check_plan_integrity(task)
            task['status'] = 'executing'
            return self._save(db, task, 'executing')

    def request_resume(self, task_id, plan_hash):
        with self.transaction() as db:
            task = self._read(db, task_id)
            if (task['status'] != 'awaiting_resume' or task.get('execution_protocol') != 1
                    or task.get('approved_hash') != plan_hash or task['plan_hash'] != plan_hash):
                raise Conflict('任务不可续跑或方案审批版本不符')
            check_plan_integrity(task)
            task.update(status='approved', error=None)
            return self._save(db, task, 'resume_requested', {'plan_hash':plan_hash})

    def record_checkpoint(self, task_id, stage, digest, *, plan_hash, adopted=False):
        with self.transaction() as db:
            task = self._read(db, task_id)
            if (task['status'] not in {'executing', 'verifying'} or task.get('execution_protocol') != 1
                    or task.get('approved_hash') != plan_hash or task['plan_hash'] != plan_hash):
                raise Conflict('当前任务不允许写入检查点')
            checkpoints = task.setdefault('checkpoints', {})
            if stage in checkpoints:
                if checkpoints[stage] != digest:
                    raise Conflict('检查点与已提交记录不一致')
                return self._save(db, task, 'stage_reused', {'stage':stage})
            checkpoints[stage] = digest
            return self._save(db, task, 'stage_adopted' if adopted else 'stage_completed', {'stage':stage})

    def _round_for_decision(self, db, task_id, plan_hash):
        task = self._read(db, task_id)
        if (task['status'] != 'awaiting_round_decision' or not task.get('series_id')
                or task['plan_hash'] != plan_hash or task.get('approved_hash') != plan_hash):
            raise Conflict('当前轮次不可决策或审批版本不符')
        check_plan_integrity(task)
        return task

    def create_replan(self, task_id, plan_hash, feedback):
        with self.transaction() as db:
            parent = self._round_for_decision(db, task_id, plan_hash)
            if parent['round_number'] >= MAX_ROUNDS:
                raise Conflict(f'本实验已达到 {MAX_ROUNDS} 轮上限')
            task = {'id':uuid.uuid4().hex, 'goal':parent['goal'], 'snapshot':parent['snapshot'],
                    'status':'created', 'created_at':now(), 'mode':parent['mode'],
                    'series_id':parent['series_id'], 'round_number':parent['round_number']+1,
                    'previous_task_id':task_id, 'previous_round':feedback,
                    'attempted_configs':parent['attempted_configs']+[parent['plan']],
                    'baseline':parent['baseline'], 'plan':None, 'plan_hash':None,
                    'diagnosis':None, 'candidate':None, 'error':None}
            if 'baseline_profile' in parent:
                task['baseline_profile'] = parent['baseline_profile']
            db.execute('INSERT INTO tasks VALUES(?,?)', (task['id'], json.dumps(task, ensure_ascii=False)))
            self._save(db, task, 'round_created', {'previous_task_id':task_id, 'round':task['round_number']})
            parent.update(status='superseded', next_task_id=task['id'])
            self._save(db, parent, 'replan_requested', {'next_task_id':task['id']})
            return task

    def request_finalization(self, task_id, plan_hash):
        with self.transaction() as db:
            task = self._round_for_decision(db, task_id, plan_hash)
            if not task['round_summary']['eligible']:
                raise Conflict('开发集门槛未通过，不能开启留出集验收')
            task.update(status='approved', finalize_requested=True)
            return self._save(db, task, 'final_candidate_frozen', {'plan_hash':plan_hash})

    def cancel(self, task_id):
        return self.transition(task_id,{'created','diagnosing','proposal_ready','awaiting_approval','awaiting_resume','approved','executing','verifying','awaiting_round_decision','awaiting_retention'},'cancelled')

    def recover_interrupted(self):
        for task in self.list():
            status = task['status']
            if task.get('execution_protocol') == 1 and status in {'approved','executing','verifying'}:
                self.transition(task['id'], {status}, 'awaiting_resume', error='服务已重启，请核对检查点后手动续跑。')
            elif status in {'created','approved','diagnosing','proposal_ready','executing','verifying','retaining'}:
                self.transition(task['id'], {status}, 'interrupted', error='服务中断；请检查已有记录后创建新任务。')
        # Keep reservations charged conservatively. A restarted process cannot
        # tell whether the provider received an in-flight request.
        with self.transaction() as db:
            db.execute("UPDATE budget SET status='uncertain' WHERE status='reserved'")

    def reserve(self, task_id, amount, cap=70., task_cap=None):
        if amount<=0:
            raise ValueError('预算预留必须为正数')
        with self.transaction() as db:
            spent=db.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM budget').fetchone()[0]
            totals = db.execute('''SELECT COALESCE(SUM(COALESCE(b.actual,b.reserved)),0), COUNT(*)
                FROM budget b LEFT JOIN tasks t ON t.id=b.task_id
                WHERE b.task_id=? OR json_extract(t.payload,'$.series_id')=(
                    SELECT json_extract(payload,'$.series_id') FROM tasks WHERE id=?)''', (task_id, task_id)).fetchone()
            task_spent, count = totals
            if task_cap is not None and task_spent+amount>task_cap:
                raise Conflict('任务或实验系列的模型预算不足，已停止新增调用')
            if count>=MAX_MODEL_CALLS:
                raise Conflict(f'任务或实验系列的模型调用已达到 {MAX_MODEL_CALLS} 次上限')
            if spent+amount>cap:
                raise Conflict('制作预算不足，已停止新增模型调用')
            call_id=uuid.uuid4().hex
            db.execute('INSERT INTO budget VALUES(?,?,?,?,?)',(call_id,task_id,amount,None,'reserved'))
            return call_id

    def settle(self, call_id, actual):
        with self.transaction() as db:
            db.execute('UPDATE budget SET actual=?,status=? WHERE id=?',(actual,'settled' if actual is not None else 'uncertain',call_id))

    def budget(self):
        with closing(self.connect()) as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM budget')]
        return {'build_cap_cny':70,'learning_reserved_cny':30,'committed_estimate_cny':sum(r['actual'] if r['actual'] is not None else r['reserved'] for r in rows),'calls':rows,'billing_note':'客户端估算；超时未确认的请求保留预留额，不等于厂商账单'}
