import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import SystemMessage

from optimizer.diagnostic_skills import select_skill
from deerflow.guardrails.provider import GuardrailRequest
from optimizer.model_policy import ScopedModelMixin, add_diagnostic_guide
from optimizer.permissions import PermissionDenied, authorize, capability_identity, run_scope, runtime_identity
from optimizer.store import Conflict
from optimizer.tool_guardrail import ScopedToolGuardrail


def runtime(binding, role):
    return SimpleNamespace(context={'thread_id':binding.thread_id, 'authz_attributes':{'ragopt_run':binding.nonce},
                                    'agent_id':role, 'is_subagent':role != 'lead'})


class PermissionsTests(unittest.TestCase):
    def test_cross_task_and_role_denials_are_audited_and_capability_expires(self):
        service = Mock()
        with run_scope(service, 'A') as binding:
            actor, role = runtime_identity(runtime(binding, 'retrieval-diagnostician'))
            token = actor.tokens[role]
            self.assertEqual(capability_identity(token), (binding, role))
            with self.assertRaises(PermissionDenied):
                authorize(actor, role, 'mcp_inspect_development', 'B', source='mcp')
            with self.assertRaises(PermissionDenied):
                authorize(actor, role, 'propose_retrieval_plan', 'A')
            self.assertEqual(service.store.event.call_count, 2)
            self.assertEqual({call.args[0] for call in service.store.event.call_args_list}, {'A'})
            service._guard.assert_not_called()
        with self.assertRaises(PermissionDenied):
            capability_identity(token)

    def test_parallel_runtime_bindings_charge_only_their_own_task(self):
        first, second = Mock(), Mock()
        class FixtureModel(ScopedModelMixin):
            def _convert_input(self, model_input):
                return SimpleNamespace(to_messages=lambda: model_input)
        def call(current):
            _, binding = FixtureModel()._prepare_ragopt_call([])
            return binding[1]
        with run_scope(first, 'A') as a, run_scope(second, 'B') as b, ThreadPoolExecutor() as pool:
            from threading import local
            state = local()
            def in_thread(binding):
                state.runtime = runtime(binding, 'lead')
                return call(binding)
            with patch('optimizer.model_policy.get_runtime', side_effect=lambda:state.runtime):
                self.assertEqual(list(pool.map(in_thread, (a,b))), ['A','B'])
        self.assertEqual(first.store.reserve.call_args.args[0], 'A')
        self.assertEqual(second.store.reserve.call_args.args[0], 'B')

    def test_skill_is_selected_by_evidence_and_visible_only_to_diagnostician(self):
        self.assertIsNone(select_skill({'baseline':{'cases':[{'scores':{'recall':1}}]}}))
        self.assertEqual(select_skill({'baseline':{'cases':[{'scores':{'recall':0.5}}]}}), 'missing-evidence')
        task = {'previous_round':{'config':{'strategy':'hybrid'},'checks':{'precision_no_regression':False}}}
        service = Mock()
        with run_scope(service, 'A', select_skill(task)) as binding:
            self.assertEqual(binding.skill, 'ranking-regression')
            for role in ['lead','retrieval-optimizer','retrieval-diagnostician']:
                messages = [SystemMessage(content='existing')]
                result = add_diagnostic_guide(messages, binding, role)
                if role == 'retrieval-diagnostician':
                    content = result[0].content
                    self.assertIn('reviewed_diagnostic_skill', content)
                    self.assertIn('ranking-regression', content)
                else:
                    self.assertIs(result, messages)
        service.store.event.assert_called_once_with('A', 'skill_injected',
                                                   {'role':'retrieval-diagnostician','skill':'ranking-regression'})

    def test_guardrail_and_model_budget_fail_closed(self):
        service = Mock()
        service.store.reserve.side_effect = Conflict('over budget')
        handler = Mock()
        class Provider:
            def invoke(self, messages):
                return handler(messages)
        class FixtureModel(ScopedModelMixin, Provider):
            def _convert_input(self, model_input):
                return SimpleNamespace(to_messages=lambda: model_input)
        with run_scope(service, 'A') as binding:
            req = GuardrailRequest('propose_retrieval_plan', {'task_id':'A'},
                                   thread_id=binding.thread_id, is_subagent=True, agent_id='retrieval-diagnostician',
                                   authz_attributes={'ragopt_run':binding.nonce})
            with patch('optimizer.tool_guardrail.get_runtime', return_value=runtime(binding, 'retrieval-diagnostician')):
                self.assertFalse(ScopedToolGuardrail().evaluate(req).allow)
            with patch('optimizer.model_policy.get_runtime', return_value=runtime(binding, 'lead')):
                with self.assertRaisesRegex(Conflict, 'over budget'):
                    FixtureModel().invoke([])
        handler.assert_not_called()
