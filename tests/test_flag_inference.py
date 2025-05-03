"""
Unit tests for the flag inference engine.
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path

import pytest

from ezstitcher.materialization.flag_engine import (
    MaterializationRule,
    RuleContext,
    FirstStepDiskInputRule,
    LastStepDiskOutputRule,
    UpstreamRequiresFSInputRule,
    DownstreamRequiresFSOutputRule,
    MaterializationRuleRegistry,
    FlagInferenceEngine
)


class TestMaterializationRule(unittest.TestCase):
    """Test the MaterializationRule interface."""
    
    def test_rule_name(self):
        """Test that rule name is derived from class name."""
        # Create a concrete rule implementation
        class TestRule(MaterializationRule):
            def applies_to(self, context):
                return True
                
            def infer_flags(self, context):
                return {}
                
        # Create an instance of the rule
        rule = TestRule()
        
        # Verify that the name is derived from the class name
        self.assertEqual(rule.name, "TestRule")
        
    def test_rule_registry(self):
        """Test the MaterializationRuleRegistry."""
        # Create a registry
        registry = MaterializationRuleRegistry()
        
        # Create some mock rules
        rule1 = MagicMock(spec=MaterializationRule)
        rule1.name = "Rule1"
        rule1.applies_to.return_value = True
        
        rule2 = MagicMock(spec=MaterializationRule)
        rule2.name = "Rule2"
        rule2.applies_to.return_value = False
        
        # Register the rules
        registry.register_rule(rule1)
        registry.register_rule(rule2)
        
        # Verify that the rules were registered
        self.assertEqual(len(registry.rules), 2)
        self.assertIn(rule1, registry.rules)
        self.assertIn(rule2, registry.rules)
        
        # Create a mock context
        context = MagicMock(spec=RuleContext)
        
        # Get applicable rules
        applicable_rules = registry.get_applicable_rules(context)
        
        # Verify that only rule1 is applicable
        self.assertEqual(len(applicable_rules), 1)
        self.assertIn(rule1, applicable_rules)
        self.assertNotIn(rule2, applicable_rules)
        
        # Verify that applies_to was called for each rule
        rule1.applies_to.assert_called_once_with(context)
        rule2.applies_to.assert_called_once_with(context)


@pytest.mark.skip(reason="Await full plan block approval")
class TestFirstStepDiskInputRule(unittest.TestCase):
    """Test the FirstStepDiskInputRule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule = FirstStepDiskInputRule()
        
    def test_applies_to(self):
        """Test that the rule applies to the first step in a pipeline."""
        # Create a context with a previous step (not first step)
        context = MagicMock(spec=RuleContext)
        context.prev_step = MagicMock()
        context.pipeline = MagicMock()
        
        # Verify that the rule does not apply
        self.assertFalse(self.rule.applies_to(context))
        
        # Create a context without a previous step (first step)
        context = MagicMock(spec=RuleContext)
        context.prev_step = None
        context.pipeline = MagicMock()
        
        # Verify that the rule applies
        self.assertTrue(self.rule.applies_to(context))
        
        # Create a context without a pipeline
        context = MagicMock(spec=RuleContext)
        context.prev_step = None
        context.pipeline = None
        
        # Verify that the rule does not apply
        self.assertFalse(self.rule.applies_to(context))
        
    def test_infer_flags_disk_input(self):
        """Test that the rule infers requires_fs_input=True for disk input."""
        # Create a context with disk input
        context = MagicMock(spec=RuleContext)
        context.processing_context.get_step_input_dir.return_value = Path("/input")
        
        # Mock is_disk_path to return True
        with patch('ezstitcher.io.path_utils.is_disk_path', return_value=True):
            # Infer flags
            flags = self.rule.infer_flags(context)
            
            # Verify that requires_fs_input is True
            self.assertTrue(flags['inferred_requires_fs_input'])
            self.assertFalse(flags['inferred_requires_fs_output'])
            
    def test_infer_flags_memory_input(self):
        """Test that the rule infers requires_fs_input=False for memory input."""
        # Create a context with memory input
        context = MagicMock(spec=RuleContext)
        context.processing_context.get_step_input_dir.return_value = MagicMock()
        
        # Mock is_disk_path to return False
        with patch('ezstitcher.io.path_utils.is_disk_path', return_value=False):
            # Infer flags
            flags = self.rule.infer_flags(context)
            
            # Verify that requires_fs_input is False
            self.assertFalse(flags['inferred_requires_fs_input'])
            self.assertFalse(flags['inferred_requires_fs_output'])


@pytest.mark.skip(reason="Await full plan block approval")
class TestLastStepDiskOutputRule(unittest.TestCase):
    """Test the LastStepDiskOutputRule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule = LastStepDiskOutputRule()
        
    def test_applies_to(self):
        """Test that the rule applies to the last step in a pipeline."""
        # Create a context with a next step (not last step)
        context = MagicMock(spec=RuleContext)
        context.next_step = MagicMock()
        context.pipeline = MagicMock()
        
        # Verify that the rule does not apply
        self.assertFalse(self.rule.applies_to(context))
        
        # Create a context without a next step (last step)
        context = MagicMock(spec=RuleContext)
        context.next_step = None
        context.pipeline = MagicMock()
        
        # Verify that the rule applies
        self.assertTrue(self.rule.applies_to(context))
        
        # Create a context without a pipeline
        context = MagicMock(spec=RuleContext)
        context.next_step = None
        context.pipeline = None
        
        # Verify that the rule does not apply
        self.assertFalse(self.rule.applies_to(context))
        
    def test_infer_flags_disk_output(self):
        """Test that the rule infers requires_fs_output=True for disk output."""
        # Create a context with disk output
        context = MagicMock(spec=RuleContext)
        context.processing_context.get_step_output_dir.return_value = Path("/output")
        
        # Mock is_disk_path to return True
        with patch('ezstitcher.io.path_utils.is_disk_path', return_value=True):
            # Infer flags
            flags = self.rule.infer_flags(context)
            
            # Verify that requires_fs_output is True
            self.assertFalse(flags['inferred_requires_fs_input'])
            self.assertTrue(flags['inferred_requires_fs_output'])
            
    def test_infer_flags_memory_output(self):
        """Test that the rule infers requires_fs_output=False for memory output."""
        # Create a context with memory output
        context = MagicMock(spec=RuleContext)
        context.processing_context.get_step_output_dir.return_value = MagicMock()
        
        # Mock is_disk_path to return False
        with patch('ezstitcher.io.path_utils.is_disk_path', return_value=False):
            # Infer flags
            flags = self.rule.infer_flags(context)
            
            # Verify that requires_fs_output is False
            self.assertFalse(flags['inferred_requires_fs_input'])
            self.assertFalse(flags['inferred_requires_fs_output'])


@pytest.mark.skip(reason="Await full plan block approval")
class TestUpstreamRequiresFSInputRule(unittest.TestCase):
    """Test the UpstreamRequiresFSInputRule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule = UpstreamRequiresFSInputRule()
        
    def test_applies_to(self):
        """Test that the rule applies to any step with a next step."""
        # Create a context without a next step
        context = MagicMock(spec=RuleContext)
        context.next_step = None
        
        # Verify that the rule does not apply
        self.assertFalse(self.rule.applies_to(context))
        
        # Create a context with a next step
        context = MagicMock(spec=RuleContext)
        context.next_step = MagicMock()
        
        # Verify that the rule applies
        self.assertTrue(self.rule.applies_to(context))
        
    def test_infer_flags_next_step_requires_fs_input(self):
        """Test that the rule infers requires_fs_output=True if next step requires fs input."""
        # Create a context with a next step that requires fs input
        context = MagicMock(spec=RuleContext)
        context.next_step = MagicMock()
        context.next_step.requires_fs_input = True
        
        # Infer flags
        flags = self.rule.infer_flags(context)
        
        # Verify that requires_fs_output is True
        self.assertFalse(flags['inferred_requires_fs_input'])
        self.assertTrue(flags['inferred_requires_fs_output'])
        
    def test_infer_flags_next_step_does_not_require_fs_input(self):
        """Test that the rule infers requires_fs_output=False if next step does not require fs input."""
        # Create a context with a next step that does not require fs input
        context = MagicMock(spec=RuleContext)
        context.next_step = MagicMock()
        context.next_step.requires_fs_input = False
        
        # Infer flags
        flags = self.rule.infer_flags(context)
        
        # Verify that requires_fs_output is False
        self.assertFalse(flags['inferred_requires_fs_input'])
        self.assertFalse(flags['inferred_requires_fs_output'])


@pytest.mark.skip(reason="Await full plan block approval")
class TestDownstreamRequiresFSOutputRule(unittest.TestCase):
    """Test the DownstreamRequiresFSOutputRule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rule = DownstreamRequiresFSOutputRule()
        
    def test_applies_to(self):
        """Test that the rule applies to any step with a previous step."""
        # Create a context without a previous step
        context = MagicMock(spec=RuleContext)
        context.prev_step = None
        
        # Verify that the rule does not apply
        self.assertFalse(self.rule.applies_to(context))
        
        # Create a context with a previous step
        context = MagicMock(spec=RuleContext)
        context.prev_step = MagicMock()
        
        # Verify that the rule applies
        self.assertTrue(self.rule.applies_to(context))
        
    def test_infer_flags_prev_step_requires_fs_output(self):
        """Test that the rule infers requires_fs_input=True if previous step requires fs output."""
        # Create a context with a previous step that requires fs output
        context = MagicMock(spec=RuleContext)
        context.prev_step = MagicMock()
        context.prev_step.requires_fs_output = True
        
        # Infer flags
        flags = self.rule.infer_flags(context)
        
        # Verify that requires_fs_input is True
        self.assertTrue(flags['inferred_requires_fs_input'])
        self.assertFalse(flags['inferred_requires_fs_output'])
        
    def test_infer_flags_prev_step_does_not_require_fs_output(self):
        """Test that the rule infers requires_fs_input=False if previous step does not require fs output."""
        # Create a context with a previous step that does not require fs output
        context = MagicMock(spec=RuleContext)
        context.prev_step = MagicMock()
        context.prev_step.requires_fs_output = False
        
        # Infer flags
        flags = self.rule.infer_flags(context)
        
        # Verify that requires_fs_input is False
        self.assertFalse(flags['inferred_requires_fs_input'])
        self.assertFalse(flags['inferred_requires_fs_output'])


@pytest.mark.skip(reason="Await full plan block approval")
class TestFlagInferenceEngine(unittest.TestCase):
    """Test the FlagInferenceEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock registry
        self.registry = MagicMock(spec=MaterializationRuleRegistry)
        
        # Create the engine with the mock registry
        self.engine = FlagInferenceEngine(registry=self.registry)
        
    def test_default_rules_registration(self):
        """Test that default rules are registered if registry is empty."""
        # Create a registry with no rules
        registry = MaterializationRuleRegistry()
        
        # Create an engine with the empty registry
        engine = FlagInferenceEngine(registry=registry)
        
        # Verify that default rules were registered
        self.assertTrue(len(registry.rules) > 0)
        
        # Verify that each rule is an instance of MaterializationRule
        for rule in registry.rules:
            self.assertIsInstance(rule, MaterializationRule)
            
    def test_infer_flags(self):
        """Test that the engine infers flags correctly."""
        # Create a step, pipeline, and context
        step = MagicMock()
        pipeline = MagicMock()
        context = MagicMock()
        
        # Create a mock rule that applies and infers flags
        rule = MagicMock(spec=MaterializationRule)
        rule.applies_to.return_value = True
        rule.infer_flags.return_value = {
            'inferred_requires_fs_input': True,
            'inferred_requires_fs_output': False
        }
        rule.name = "MockRule"
        
        # Set up the registry to return the mock rule
        self.registry.get_applicable_rules.return_value = [rule]
        
        # Infer flags
        with patch('ezstitcher.materialization.flag_engine.RuleContext.from_pipeline_position') as mock_from_pipeline_position:
            mock_from_pipeline_position.return_value = MagicMock(spec=RuleContext)
            flags = self.engine.infer_flags(step, pipeline, context)
            mock_from_pipeline_position.assert_called_once_with(step, pipeline, context)
        
        # Verify that the registry was used to get applicable rules
        self.registry.get_applicable_rules.assert_called_once()
        
        # Verify that the rule was used to infer flags
        rule.infer_flags.assert_called_once()
        
        # Verify that the inferred flags are correct
        self.assertTrue(flags['inferred_requires_fs_input'])
        self.assertFalse(flags['inferred_requires_fs_output'])
        
    def test_infer_flags_multiple_rules(self):
        """Test that the engine merges flags from multiple rules correctly."""
        # Create a step, pipeline, and context
        step = MagicMock()
        pipeline = MagicMock()
        context = MagicMock()
        
        # Create mock rules that apply and infer different flags
        rule1 = MagicMock(spec=MaterializationRule)
        rule1.applies_to.return_value = True
        rule1.infer_flags.return_value = {
            'inferred_requires_fs_input': True,
            'inferred_requires_fs_output': False
        }
        rule1.name = "Rule1"
        
        rule2 = MagicMock(spec=MaterializationRule)
        rule2.applies_to.return_value = True
        rule2.infer_flags.return_value = {
            'inferred_requires_fs_input': False,
            'inferred_requires_fs_output': True
        }
        rule2.name = "Rule2"
        
        # Set up the registry to return both mock rules
        self.registry.get_applicable_rules.return_value = [rule1, rule2]
        
        # Infer flags
        with patch('ezstitcher.materialization.flag_engine.RuleContext.from_pipeline_position') as mock_from_pipeline_position:
            mock_from_pipeline_position.return_value = MagicMock(spec=RuleContext)
            flags = self.engine.infer_flags(step, pipeline, context)
            mock_from_pipeline_position.assert_called_once_with(step, pipeline, context)
        
        # Verify that both rules were used to infer flags
        rule1.infer_flags.assert_called_once()
        rule2.infer_flags.assert_called_once()
        
        # Verify that the inferred flags are merged correctly (True takes precedence)
        self.assertTrue(flags['inferred_requires_fs_input'])
        self.assertTrue(flags['inferred_requires_fs_output'])


@pytest.mark.skip(reason="Await full plan block approval")
class TestFlagInferenceIntegration(unittest.TestCase):
    """Integration tests for flag inference."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock orchestrator
        self.orchestrator = MagicMock()
        self.orchestrator.storage_mode = "memory"
        self.orchestrator.overlay_mode = "auto"
        
        # Create a mock materialization manager
        self.materialization_manager = MagicMock()
        self.orchestrator.materialization_manager = self.materialization_manager
        
        # Create a mock flag inference engine
        self.flag_inference_engine = MagicMock(spec=FlagInferenceEngine)
        self.orchestrator.flag_inference_engine = self.flag_inference_engine
        
    def test_pipeline_with_inferred_flags(self):
        """Test a pipeline with inferred flags."""
        from ezstitcher.core.pipeline import Pipeline
        from ezstitcher.core.steps import Step
        
        # Create a pipeline with three steps
        step1 = Step(func=lambda x: x, name="Step 1")
        step2 = Step(func=lambda x: x, name="Step 2")
        step3 = Step(func=lambda x: x, name="Step 3")
        pipeline = Pipeline(steps=[step1, step2, step3], name="Test Pipeline")
        
        # Create a mock context
        context = MagicMock()
        context.orchestrator = self.orchestrator
        context.well_filter = ["A01"]
        
        # Configure the flag inference engine to return specific flags
        self.flag_inference_engine.infer_flags.side_effect = lambda step, pipeline, context: {
            'inferred_requires_fs_input': step == step1,  # First step needs input
            'inferred_requires_fs_output': step == step3   # Last step needs output
        }
        
        # Configure MaterializationResolver.needs_materialization to return True for step1 and step3
        with patch('ezstitcher.io.materialization_resolver.MaterializationResolver.needs_materialization') as mock_needs_materialization:
            mock_needs_materialization.side_effect = lambda step, manager, context, pipeline: step in [step1, step3]
            
            # Run the pipeline
            pipeline.run(context)
            
            # Verify that the engine was used to infer flags
            self.assertEqual(self.flag_inference_engine.infer_flags.call_count, 3)
            
            # Verify that materialization was triggered for first and last steps
            self.materialization_manager.prepare_for_step.assert_any_call(step1, "A01", ANY)
            self.materialization_manager.prepare_for_step.assert_any_call(step3, "A01", ANY)
            
            # Verify that step instances were not modified
            self.assertFalse(getattr(step1, 'requires_fs_input', False))
            self.assertFalse(getattr(step1, 'requires_fs_output', False))
            self.assertFalse(getattr(step3, 'requires_fs_input', False))
            self.assertFalse(getattr(step3, 'requires_fs_output', False))
