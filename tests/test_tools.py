import unittest
from optimizer.tools import propose_retrieval_plan


class ToolSchemaTests(unittest.TestCase):
    def test_model_run_is_blocked_without_scoped_model(self):
        from optimizer.model_policy import require_scoped_model
        with self.assertRaisesRegex(RuntimeError,'任务绑定与预算控制'):
            require_scoped_model(object)

    def test_retrieval_configuration_is_present_in_model_tool_schema(self):
        self.assertIn('retrieval_config',propose_retrieval_plan.args)
        self.assertNotIn('config',propose_retrieval_plan.args)
        schema = propose_retrieval_plan.tool_call_schema.model_json_schema()
        self.assertIn('experiment', schema['required'])
        self.assertNotIn('runtime', schema['properties'])
