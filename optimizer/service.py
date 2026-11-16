"""Experiment orchestration. Model tools cannot approve their own changes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, os, shutil, threading, traceback, subprocess
from .isolation import confined_path, snapshot_hash
from .retrieval import build_chunks, evaluate, validate_config, worker_python
from .store import Store, Conflict
from .evaluation import case_regressions, aggregate, score_case
from .artifacts import freeze_adapter, candidate_hash, write_chunks, record_diff
from .experiment_plan import ExperimentSpec, assess_experiment, build_experiment
from .checkpoints import STAGES, read_json, run_stage, stage_directory, verify_stage, write_json
from .rounds import development_feedback
from .memory import SessionMemory

OVERLAP_BASELINE = {'chunking':'fixed', 'chunk_size':64}
RRF_BASELINE = {'strategy':'hybrid'}
EXPERIMENT_BASELINES = {'fixed_overlap':OVERLAP_BASELINE, 'weighted_rrf':RRF_BASELINE}


class OptimizerService:
    def __init__(self, root: Path, evaluator=evaluate):
        self.root=root.resolve()
        self.store=Store(root/'runtime/optimizer.db')
        self.memory=SessionMemory(self.store)
        self.owner = self.store.acquire_owner()
        self.evaluator=evaluator
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='ragopt')
        self.agent_lock=threading.Lock()
        try:
            self.store.recover_interrupted()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        try:
            self.memory.close()
        finally:
            self.owner.close()

    def add_memory_note(self, task_id, text, *, include_in_model=False):
        text = text.strip()
        if not 1 <= len(text) <= 600:
            raise ValueError('备注须为 1–600 字')
        self.store.event(task_id, 'series_memory_note', {'text': text, 'include_in_model': include_in_model})
        return self.memory.view(task_id)

    def load(self, name):
        if name not in {'corpus','original-corpus','development','regression','heldout'}:
            raise ValueError('未知数据资源')
        return json.loads((self.root/f'data/{name}.json').read_text(encoding='utf-8'))

    def model_config_path(self):
        configured = os.environ.get('DEER_FLOW_CONFIG_PATH')
        return Path(configured) if configured else self.root/'config.yaml'

    def fingerprint(self):
        # Labels/splits and baseline source are covered, runtime artifacts are not.
        content=snapshot_hash(self.root/'baseline/rag')+snapshot_hash(self.root/'data')
        if (self.root/'optimizer').exists():
            content+=snapshot_hash(self.root/'optimizer')
        for path in (self.model_config_path(), self.root/'extensions_config.json'):
            if path.exists():
                content += hashlib.sha256(path.read_bytes()).hexdigest()
        graph_check = self.root/'scripts/check_candidate.py'
        if graph_check.exists():
            content += hashlib.sha256(graph_check.read_bytes()).hexdigest()
        return hashlib.sha256(content.encode()).hexdigest()

    def work(self, task_id):
        self.store.get(task_id)
        parent=confined_path(self.root,'experiments')
        parent.mkdir(exist_ok=True)
        path=confined_path(parent,task_id)
        path.mkdir(exist_ok=True)
        return path

    def _guard(self, task_id):
        task=self.store.get(task_id)
        if task['status'] in {'cancelled','rejected','interrupted','failed','superseded'}:
            raise Conflict('任务已停止')
        if task['snapshot']!=self.fingerprint():
            raise Conflict('源代码或数据集已变化；请创建新任务')
        return task

    def _background(self, task_id, function):
        try:
            function(task_id)
        except BaseException as exc:
            task=self.store.get(task_id)
            if task['status'] not in {'cancelled','failed','interrupted'}:
                self.store.transition(task_id,{task['status']},'failed',error=f'{type(exc).__name__}: {str(exc)[:500]}')
            # Raw provider exceptions can contain credentials: persist only the
            # sanitized exception class and local traceback frames.
            self.store.event(task_id,'failure',{'type':type(exc).__name__,'frames':traceback.format_tb(exc.__traceback__)[-5:]})

    def create(self, goal: str, run=True, mode='deerflow', *, iterative=False, overlap_experiment=False, rrf_experiment=False):
        if mode not in {'deerflow','offline'}:
            raise ValueError('未知运行模式')
        if not isinstance(goal,str) or not 1<=len(goal.strip())<=2000:
            raise ValueError('请填写 1–2000 字的诊断目标')
        if overlap_experiment and rrf_experiment:
            raise ValueError('一次任务只选择一种专项对照')
        task=self.store.create(goal.strip(),self.fingerprint(),iterative=iterative)
        profile = 'weighted_rrf' if rrf_experiment else 'fixed_overlap' if overlap_experiment else None
        updates = {'baseline_profile':profile} if profile else {}
        self.store.transition(task['id'],{'created'},'created',mode=mode,**updates)
        if run:
            self._queue_diagnosis(task['id'], mode)
        return self.store.get(task['id'])

    def _queue_diagnosis(self, task_id, mode):
        function = self.diagnose_deerflow if mode == 'deerflow' else self.diagnose_offline
        self.pool.submit(self._background, task_id, function)

    def _baseline(self, task_id):
        task = self._guard(task_id)
        self.store.transition(task_id,{'created'},'diagnosing')
        if task.get('previous_task_id') and task.get('baseline'):
            self.store.event(task_id, 'baseline_reused', {'series_id':task['series_id']})
            return task['baseline']
        config = EXPERIMENT_BASELINES.get(task.get('baseline_profile'), {})
        self.store.event(task_id,'evaluation_started',{'split':'development','config':validate_config(config)})
        report=self.evaluator(self.root,self.load('corpus'),self.load('development'),config,self.work(task_id)/'baseline','development')
        self._guard(task_id)
        self.store.transition(task_id,{'diagnosing'},'diagnosing',baseline=report)
        self.store.event(task_id,'evaluation_completed',{'metrics':report['metrics']})
        return report

    def diagnose_offline(self, task_id):
        report=self._baseline(task_id)
        failures=[{'id':r['id'],'question':r['question'],'recall':r['scores']['recall']} for r in report['cases'] if r['scores']['recall'] is not None and r['scores']['recall']<1]
        diagnosis={'mode':'offline','note':'固定统计分析；不是 DeerFlow 模型诊断','failures':failures,'evidence':'仅依据开发集检索结果，不推断原因已被证明'}
        self.store.transition(task_id,{'diagnosing'},'diagnosing',diagnosis=diagnosis)
        if not failures:
            raise ValueError('开发集未发现可用于本次实验的漏召回题')
        task = self.store.get(task_id)
        feedback = task.get('previous_round')
        targets = {row['id'] for row in (feedback or {}).get('case_deltas', []) if row['target']}
        target = next((row for row in failures if row['id'] in targets), failures[0])
        if task.get('baseline_profile') == 'fixed_overlap':
            return self.propose_overlap(task, failures, feedback)
        if task.get('baseline_profile') == 'weighted_rrf':
            return self.propose_rrf(task, failures, feedback)
        choices = [({'strategy':'hybrid'}, '词面信号可能补足向量排序的漏召回'),
                   ({'strategy':'bm25'}, '单独使用词面检索可以验证混合排序是否稀释关键词信号'),
                   ({'chunking':'section'}, '按章节保留相邻条款可能补全组合问题的证据')]
        from .query_split import split_query
        attempted = [validate_config(tried) for tried in task.get('attempted_configs', [])]
        split_targets = [row for row in failures if len(split_query(row['question'])) > 1]
        # Replanning still needs a reason tied to the previous round's evidence.
        eligible_splits = [row for row in split_targets
                           if not feedback or any(delta['id'] == row['id'] for delta in feedback['case_deltas'])]
        if eligible_splits and validate_config({'query_split':True}) not in attempted:
            target = next((row for row in eligible_splits if row['id'] in targets), eligible_splits[0])
            choices.insert(0, ({'query_split':True}, '拆分多条件问题后分别召回，可能补足整句检索遗漏的条款'))
        config, hypothesis = next((config, hypothesis) for config, hypothesis in choices
                                  if validate_config(config) not in attempted)
        reason = '离线单变量候选，批准后对照评测。'
        if feedback:
            row = next((row for row in feedback['case_deltas'] if row['id'] == target['id']), None)
            if row:
                reason = (f"上一轮开发集 {row['id']} 的证据召回为 {row['candidate_recall']:.1%}，"
                          f"基线为 {row['baseline_recall']:.1%}。本轮改变一个配置项检验新假设。")
        experiment = ExperimentSpec(hypothesis=hypothesis,
                                    target_case_ids=[target['id']],
                                    min_recall_gain=min(0.1, 1 - target['recall']))
        return self.propose(task_id, config, reason, experiment)

    def propose_overlap(self, task, failures, feedback):
        attempted = [validate_config(tried) for tried in task.get('attempted_configs', [])]
        reference = validate_config(OVERLAP_BASELINE)
        config = next((reference | {'chunk_overlap': value} for value in (16, 24, 32)
                       if reference | {'chunk_overlap': value} not in attempted), None)
        if config is None:
            raise Conflict('已测试所有预定重叠量')
        prior_ids = {row['id'] for row in (feedback or {}).get('case_deltas', [])}
        baseline_rows = {row['id']: row for row in task['baseline']['cases']}

        def partial_evidence(failure):
            row = baseline_rows[failure['id']]
            required = {item for group in row['required'] for item in group}
            return any(span['id'] in required and
                       (span['covered_start'], span['covered_end']) != (span['start'], span['end'])
                       for hit in row['results'] for span in hit.get('evidence_spans', []))

        eligible = [row for row in failures if not feedback or row['id'] in prior_ids]
        if not eligible:
            raise Conflict('上一轮没有可用于重规划的漏召回证据')
        target = next((row for row in eligible if partial_evidence(row)), eligible[0])
        experiment = ExperimentSpec(hypothesis='定长窗口加入重叠后可能保留完整跨边界条款',
                                    target_case_ids=[target['id']], min_recall_gain=min(.1, 1-target['recall']))
        reason = f"开发集 {target['id']} 证据召回 {target['recall']:.1%}，比较相同窗口长度下的覆盖变化。"
        return self.propose(task['id'], config, reason, experiment)

    def propose_rrf(self, task, failures, feedback):
        reference = validate_config(RRF_BASELINE)
        attempted = [validate_config(config) for config in task.get('attempted_configs', [])]
        prior = {row['id'] for row in (feedback or {}).get('case_deltas', [])}
        failing_ids = {row['id'] for row in failures}
        rows = [row for row in task['baseline']['cases']
                if row['id'] in failing_ids and (not feedback or row['id'] in prior)]
        chunks = {chunk['id']:chunk for chunk in build_chunks(self.load('corpus'), reference)}
        for weight, route in ((75,'vector'), (25,'bm25'), (65,'vector')):
            config = reference | {'rrf_vector_pct':weight}
            if config in attempted:
                continue
            for row in rows:
                hits = [chunks[chunk_id] for chunk_id in row['rrf_trace'][route]]
                recall = row['scores']['recall']
                if score_case(row, hits, reference['top_k'])['recall'] <= recall:
                    continue
                experiment = ExperimentSpec(hypothesis=f'提高 {route} 的融合权重可能找回掉出 Top-K 的证据',
                                            target_case_ids=[row['id']], min_recall_gain=min(.1, 1-recall))
                reason = f"开发集 {row['id']} 的 {route} 单路 Top-K 证据召回高于等权融合。"
                return self.propose(task['id'], config, reason, experiment)
        raise Conflict('开发集暂无符合条件且尚未尝试的 RRF 权重方案')

    def diagnose_deerflow(self, task_id):
        self._baseline(task_id)
        self.store.transition(task_id,{'diagnosing'},'diagnosing',agent_running=True)
        from .agent import run_diagnosis
        with self.agent_lock:
            self._guard(task_id)
            result=run_diagnosis(self,task_id)
        self._guard(task_id)
        task=self.store.get(task_id)
        if task['status']!='proposal_ready':
            if '[TOKEN BUDGET EXCEEDED]' in result:
                raise RuntimeError('DeerFlow 诊断达到本轮 token 上限，未提交方案；请查看任务轨迹')
            raise RuntimeError('DeerFlow 已结束但未提交可审批方案；请查看任务轨迹')
        self.store.transition(task_id,{'proposal_ready'},'awaiting_approval',diagnosis={'mode':'deerflow','summary':result},agent_running=False)

    def agent_view(self, task_id):
        task=self._guard(task_id)
        report=task.get('baseline') or {}
        rows=report.get('cases',[])
        failures=[r for r in rows if r['scores']['recall'] is not None and r['scores']['recall']<1]
        unknown=[r for r in rows if r['scores']['recall'] is None]
        examples=[r for r in rows if r['scores']['recall']==1][:2]
        selected=list({r['id']:r for r in failures[:8]+unknown[:2]+examples}.values())
        # Never return held-out questions/labels, credential files or raw logs.
        return {'id':task_id,'goal':task['goal'],'status':task['status'],'snapshot':task['snapshot'],
                'metrics':report.get('metrics'),'config':report.get('config'),
                'development_cases':selected,'development_case_count':len(rows),'omitted_case_count':len(rows)-len(selected),
                'case_selection':'优先漏召回案例，再给无答案题和正常对照；完整报告保留在本地',
                'plan':task.get('plan'),'reason':task.get('reason'),'experiment_plan':task.get('experiment_plan'),
                'previous_round':task.get('previous_round'),'attempted_configs':task.get('attempted_configs', []),
                'allowed_configs':{'strategy':['vector','bm25','hybrid'],'chunking':['paragraph','section','fixed'],
                                   'top_k':'1–10','chunk_size':'16–1000','context_chars':'100–6000','rerank':'默认 false；启用前确认缓存可用',
                                   'query_split':'默认 false；多条件漏召回可试 true，仅纯向量且关闭重排',
                                   'chunk_overlap':'默认 0；选择定长重叠实验后只修改此项，小于窗口长度',
                                   'rrf_vector_pct':'等权为 50；RRF 专项只改此项，取 10–90 整数，BM25 为 100 减该值'},
                'warning':'没有基线证据时不要宣称根因；不可访问留出题或批准自己的修改'}

    def sources(self, task_id, document_ids):
        task=self._guard(task_id)
        if len(document_ids)>8:
            raise ValueError('一次最多读取 8 篇政策')
        return [d for d in self.load('corpus') if d['id'] in document_ids]

    def propose(self, task_id, config, reason, experiment=None):
        task=self._guard(task_id)
        if not isinstance(reason,str) or not 2<=len(reason)<=6000:
            raise ValueError('需要说明方案依据及不确定性')
        config=validate_config(config)
        if task.get('baseline_profile') == 'fixed_overlap':
            reference = validate_config(OVERLAP_BASELINE)
            if experiment is None or not config['chunk_overlap'] or config | {'chunk_overlap':0} != reference:
                raise ValueError('重叠切分实验必须以定长无重叠为对照，仅提交正整数重叠量')
        elif config['chunk_overlap']:
            raise ValueError('请从定长重叠实验入口建立无重叠对照')
        if task.get('baseline_profile') == 'weighted_rrf':
            reference = validate_config(RRF_BASELINE)
            if experiment is None or config['rrf_vector_pct'] == 50 or config | {'rrf_vector_pct':50} != reference:
                raise ValueError('RRF 权重实验只允许在等权混合基线上改权重')
        elif config['rrf_vector_pct'] != 50:
            raise ValueError('请从 RRF 权重实验入口建立等权对照')
        if config in [validate_config(tried) for tried in task.get('attempted_configs', [])]:
            raise ValueError('该配置已在本实验系列验证，请提出不同的单变量方案')
        if task.get('series_id') and experiment is None:
            raise ValueError('多轮实验需要可校验的结构化计划')
        previous = task.get('previous_round')
        if previous and not any(row['id'] in reason for row in previous['case_deltas']):
            raise ValueError('重规划理由须引用上一轮开发集差异中的至少一个题 ID')
        if config['rerank'] and not (self.root/'.cache/models/models--BAAI--bge-reranker-base').exists():
            raise ValueError('重排序模型缓存未就绪；不得静默联网下载')
        structured = None
        if experiment is not None:
            baseline = task.get('baseline')
            if not baseline:
                raise ValueError('提交实验前必须完成开发集基线评测')
            reference_config = validate_config(baseline.get('config') or {})
            structured = build_experiment(ExperimentSpec.model_validate(experiment), baseline, config, reference_config)
        return self.store.propose(task_id,config,reason,pending=bool(task.get('agent_running')),
                                  experiment_plan=structured)

    def approve(self, task_id, plan_hash, run=True):
        self._guard(task_id)
        task=self.store.approve(task_id,plan_hash,resumable=True)
        if run:
            self.pool.submit(self._background,task_id,self.execute)
        return task

    def resume(self, task_id, plan_hash, run=True):
        self._guard(task_id)
        task = self.store.request_resume(task_id, plan_hash)
        if run:
            self.pool.submit(self._background, task_id, self.execute)
        return task

    def replan(self, task_id, plan_hash, run=True):
        task = self._guard(task_id)
        if task['status'] != 'awaiting_round_decision':
            raise Conflict('当前轮次没有待决策的开发集结果')
        stage = stage_directory(self.work(task_id), 'development')
        payload, _ = verify_stage(stage, task, 'development')
        feedback = development_feedback(task, read_json(confined_path(payload, 'report.json')))
        new_task = self.store.create_replan(task_id, plan_hash, feedback)
        if run:
            self._queue_diagnosis(new_task['id'], new_task['mode'])
        return new_task

    def finalize(self, task_id, plan_hash, run=True):
        self._guard(task_id)
        task = self.store.request_finalization(task_id, plan_hash)
        if run:
            self.pool.submit(self._background, task_id, self.execute)
        return task

    def execute(self, task_id):
        self._guard(task_id)
        task=self.store.claim_execution(task_id)
        config=validate_config(task['plan'])
        work=self.work(task_id)
        if confined_path(work, 'candidate').exists():
            raise Conflict('候选目录已存在；不覆盖或重放执行')
        built = run_stage(self, task_id, work, 'build', lambda output: self._build_candidate(output, config))
        self.store.transition(task_id, {'executing'}, 'verifying')
        development = run_stage(self, task_id, work, 'development',
                                lambda output: self._develop_candidate(output, built/'candidate', config))
        candidate = confined_path(development, 'candidate')
        dev = read_json(confined_path(development, 'report.json'))
        if task.get('series_id') and not task.get('finalize_requested'):
            summary = development_feedback(task, dev)
            self._guard(task_id)
            self.store.transition(task_id, {'verifying'}, 'awaiting_round_decision', round_summary=summary)
            return summary
        verification = run_stage(self, task_id, work, 'verification',
                                 lambda output: self._verify_candidate(output, candidate, dev, task, config))
        comparison = read_json(confined_path(verification, 'comparison.json'))
        if candidate_hash(candidate) != comparison['candidate_hash']:
            raise Conflict('最终候选和已验证指纹不一致')
        self._guard(task_id)
        self.store.transition(task_id, {'verifying'}, 'awaiting_retention', candidate=comparison)
        event = {'development':dev['metrics'], 'heldout':comparison['heldout']}
        if 'experiment_assessment' in comparison:
            event['experiment_assessment'] = comparison['experiment_assessment']
        self.store.event(task_id, 'verification_completed', event)
        return comparison

    def _build_candidate(self, work, config):
        candidate=confined_path(work,'candidate')
        shutil.copytree(self.root/'baseline/rag',candidate)
        (candidate/'retrieval_config.json').write_text(json.dumps(config|{'model_cache':str(self.root/'.cache/models')},ensure_ascii=False,indent=2),encoding='utf-8')
        write_chunks(candidate,self.load('corpus'),config)
        freeze_adapter(self.root,candidate,query_split=config['query_split'])
        replacement='''"""Generated bounded adapter: see approved retrieval_config.json."""
from pathlib import Path
from candidate_retrieval.adapter import load_resources, rank_chunks, format_evidence

def load_retriever():
    return load_resources(Path(__file__).parent)
'''
        (candidate/'retrieval.py').write_text(replacement,encoding='utf-8')
        # Preserve the original graph and change only candidate selection sizes.
        graph_path=candidate/'rag_graph.py'
        if graph_path.exists():
            graph=graph_path.read_text(encoding='utf-8')
            graph=graph.replace('selected = ranked[:2]',f"selected = ranked[:{config['top_k']}]")
            graph=graph.replace('selected = state["ranked_chunks"][:4]',f"selected = state[\"ranked_chunks\"][:{min(config['top_k']+2,10)}]")
            graph_path.write_text(graph,encoding='utf-8')
        (candidate/'OPTIMIZER-README.md').write_text('此目录为经批准的隔离候选，不替换原 RAG。使用主项目 scripts/start.ps1 查看任务和报告。retrieval_config.json 控制检索，启动时按 chunks.json 指纹复用或重建向量索引。原学习脚本仅作来源保留，不要运行旧 chunk_practice.py 覆盖本候选语料。',encoding='utf-8')
        record_diff(self.root/'baseline/rag',candidate,confined_path(work,'candidate.patch'))

    def _develop_candidate(self, work, built, config):
        candidate = confined_path(work, 'candidate')
        frozen_hash = candidate_hash(built, include_index=False)
        shutil.copytree(built, candidate)
        report = self.evaluator(self.root, self.load('corpus'), self.load('development'), config,
                                work/'evaluation', 'development', candidate_path=candidate)
        if candidate_hash(candidate, include_index=False) != frozen_hash:
            raise Conflict('开发集评测期间候选源文件发生变化')
        write_json(confined_path(work, 'report.json'), report)

    def _verify_candidate(self, work, developed, dev, task, config):
        task_id = task['id']
        candidate = confined_path(work, 'verification-candidate')
        indexed_hash = candidate_hash(developed)
        shutil.copytree(developed, candidate)
        frozen_hash=candidate_hash(developed,include_index=False)
        self._guard(task_id)
        held=self.evaluator(self.root,self.load('corpus'),self.load('heldout'),config,work/'candidate-heldout','heldout',candidate_path=candidate)
        self._guard(task_id)
        # Baseline held-out comparison is computed only after the plan is frozen.
        reference = EXPERIMENT_BASELINES.get(task.get('baseline_profile'), {})
        held_baseline=self.evaluator(self.root,self.load('corpus'),self.load('heldout'),reference,work/'baseline-heldout','heldout')
        regression_candidate=confined_path(work,'regression-candidate')
        shutil.copytree(candidate,regression_candidate)
        write_chunks(regression_candidate,self.load('original-corpus'),config)
        reg_baseline=self.evaluator(self.root,self.load('original-corpus'),self.load('regression'),{},work/'regression-baseline','original-regression')
        reg_candidate=self.evaluator(self.root,self.load('original-corpus'),self.load('regression'),config,work/'regression-result','original-regression',candidate_path=regression_candidate)
        if candidate_hash(candidate,include_index=False)!=frozen_hash or candidate_hash(candidate)!=indexed_hash:
            raise Conflict('评测期间候选文件发生变化，结果作废')
        self._guard(task_id)
        env=os.environ.copy()
        env.update(PYTHONPATH=str(self.root),PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        graph_work=confined_path(work,'graph-check')
        graph_work.mkdir(exist_ok=False)
        graph_output=confined_path(graph_work,'result.json')
        graph_error=confined_path(graph_work,'error.log')
        smoke=subprocess.run([str(worker_python(self.root)),str(self.root/'scripts/check_candidate.py'),str(candidate),str(graph_output)],
                             env=env,cwd=self.root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        if smoke.returncode:
            graph_error.write_text(smoke.stderr[-16000:],encoding='utf-8')
            raise RuntimeError('实际候选 RAG 图检查失败，见 graph-check/error.log')
        self._guard(task_id)
        # Only aggregate held-out metrics enter task state.
        comparison={'development':dev,'heldout':{'metrics':held['metrics'],'baseline_metrics':held_baseline['metrics']},
                    'candidate_path':str(developed),'candidate_hash':indexed_hash,'candidate_source_hash':frozen_hash,
                    'graph_check':json.loads(graph_output.read_text(encoding='utf-8')),
                    'answer_quality':'待真实原 RAG 回答抽查；未宣称回答质量改善',
                    'original_regression':{'baseline':reg_baseline['metrics'],'candidate':reg_candidate['metrics'],
                                           'case_regressions':case_regressions(reg_baseline,reg_candidate),
                                           'scope':'原语料上的真实检索对照；不等于真实模型回答质量验证'},
                    'retention_checks':{
                        'heldout_recall_no_regression':held['metrics']['recall']>=held_baseline['metrics']['recall'],
                        'development_recall_no_regression':dev['metrics']['recall']>=task['baseline']['metrics']['recall'],
                        'precision_no_regression':dev['metrics']['precision']>=task['baseline']['metrics']['precision'],
                        'original_retrieval_no_regression':not case_regressions(reg_baseline,reg_candidate)}}
        if task.get('baseline_profile') == 'fixed_overlap':
            fixed_reg=self.evaluator(self.root,self.load('original-corpus'),self.load('regression'),reference,
                                     work/'regression-fixed-baseline','original-regression')
            comparison['original_regression']['fixed_baseline'] = fixed_reg['metrics']
            comparison['original_regression']['fixed_case_regressions'] = case_regressions(fixed_reg,reg_candidate)
        if task.get('baseline_profile') == 'weighted_rrf':
            vector_dev = self.evaluator(self.root,self.load('corpus'),self.load('development'),{},
                                        work/'vector-development','development')['metrics']
            vector_held = self.evaluator(self.root,self.load('corpus'),self.load('heldout'),{},
                                         work/'vector-heldout','heldout')['metrics']
            comparison['vector_baseline'] = {'development':vector_dev, 'heldout':vector_held}
            comparison['retention_checks']['vector_development_no_regression'] = (
                dev['metrics']['recall'] >= vector_dev['recall'] and dev['metrics']['precision'] >= vector_dev['precision'])
            comparison['retention_checks']['vector_heldout_no_regression'] = held['metrics']['recall'] >= vector_held['recall']
        if candidate_hash(candidate)!=indexed_hash:
            raise Conflict('接口检查期间候选源文件发生变化')
        if 'experiment_plan' in task:
            assessment = assess_experiment(task['experiment_plan'], dev)
            comparison['experiment_assessment'] = assessment
            comparison['retention_checks']['approved_target_recall_gain'] = assessment['passed']
        write_json(confined_path(work, 'comparison.json'), comparison)

    def replay(self, task_id):
        task, events = self.store.read_trace(task_id)
        timeline = [{'seq':event['seq'], 'at':event['at'], 'kind':event['kind'],
                     **({'stage':event['payload']['stage']} if 'stage' in event['payload'] else {})}
                    for event in events]
        comparison = task.get('candidate') or {}
        report = comparison.get('development')
        checkpoints = {}
        if task.get('execution_protocol') == 1:
            report = None
            experiments = confined_path(self.root, 'experiments')
            work = confined_path(experiments, task_id) if experiments.is_dir() else None
            for stage in STAGES:
                try:
                    path = stage_directory(work, stage) if work and work.is_dir() else None
                    if not path or not path.exists():
                        checkpoints[stage] = {'state':'missing' if stage in task.get('checkpoints', {}) else 'pending'}
                        continue
                    payload, digest = verify_stage(path, task, stage)
                    if stage == 'development':
                        report = read_json(confined_path(payload, 'report.json'))
                except (ValueError, OSError) as exc:
                    checkpoints[stage] = {'state':'invalid', 'reason':str(exc)}
                    continue
                state = 'verified' if stage in task.get('checkpoints', {}) else 'published'
                checkpoints[stage] = {'state':state, 'digest':digest}
        metrics = None
        if report and report.get('cases') and all('required' in row and 'results' in row for row in report['cases']):
            config = validate_config(report['config'])
            rescored = aggregate([score_case(row, row['results'], config['top_k']) for row in report['cases']])
            metrics = {'saved':report['metrics'], 'rescored':rescored, 'matches':report['metrics'] == rescored}
        return {'task_id':task_id, 'status':task['status'], 'timeline':timeline, 'checkpoints':checkpoints,
                'development_metrics':metrics, 'model_calls':0}

    def retain(self, task_id, keep: bool):
        self._guard(task_id)
        task=self.store.get(task_id)
        if task['status']!='awaiting_retention':
            raise Conflict('任务尚未到保留决定阶段')
        if not keep:
            return self.store.transition(task_id,{'awaiting_retention'},'rejected')
        work = self.work(task_id)
        relative = 'stages/development/payload/candidate' if task.get('execution_protocol') == 1 else 'candidate'
        candidate=confined_path(work, relative)
        if not all(task['candidate'].get('retention_checks',{}).values()) or not task['candidate'].get('retention_checks'):
            raise Conflict('候选未满足保留门槛，请查看计划目标与回归结果')
        if candidate_hash(candidate)!=task['candidate']['candidate_hash']:
            raise Conflict('候选文件已变化，不能保留未经验证的版本')
        self.store.transition(task_id,{'awaiting_retention'},'retaining')
        retained=confined_path(self.root,'retained')
        retained.mkdir(exist_ok=True)
        destination=confined_path(retained,task_id)
        try:
            shutil.copytree(candidate,destination)
            if candidate_hash(destination)!=task['candidate']['candidate_hash'] or candidate_hash(candidate)!=task['candidate']['candidate_hash']:
                raise Conflict('保留复制期间候选发生变化')
            return self.store.transition(task_id,{'retaining'},'retained',retained_path=str(destination))
        except Exception:
            self.store.transition(task_id,{'retaining'},'failed',error='保留复制或完整性验证失败；未修改原 RAG')
            raise

    def reject(self, task_id):
        return self.store.transition(task_id,{'awaiting_approval'},'rejected')


_service=None


def get_service():
    global _service
    if _service is None:
        _service=OptimizerService(Path(os.environ.get('RAGOPT_ROOT',Path(__file__).resolve().parents[1])))
    return _service
