"""
Flag inference engine for EZStitcher.

This module provides a rule-based engine for inferring materialization flags
based on pipeline structure and step type, without modifying the immutable
step classes.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path

from ezstitcher.io.path_utils import is_disk_path

logger = logging.getLogger(__name__)


class MaterializationRule(ABC):
    """Interface for materialization flag inference rules."""
    
    @abstractmethod
    def applies_to(self, context: 'RuleContext') -> bool:
        """
        Determine if this rule applies to the given context.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            True if the rule applies, False otherwise
        """
        pass
        
    @abstractmethod
    def infer_flags(self, context: 'RuleContext') -> Dict[str, bool]:
        """
        Infer materialization flags based on the rule logic.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            Dictionary of inferred flags
        """
        pass
        
    @property
    def name(self) -> str:
        """Get the name of the rule."""
        return self.__class__.__name__


@dataclass
class RuleContext:
    """Context for materialization rule evaluation."""
    
    step: Any
    pipeline: Optional[Any] = None
    processing_context: Optional[Any] = None
    prev_step: Optional[Any] = None
    next_step: Optional[Any] = None
    
    @classmethod
    def from_pipeline_position(cls, step: Any, pipeline: Any, processing_context: Optional[Any] = None) -> 'RuleContext':
        """
        Create a context from a step's position in a pipeline.
        
        Args:
            step: The step being evaluated
            pipeline: The pipeline containing the step
            processing_context: The processing context
            
        Returns:
            A RuleContext object
        """
        if not pipeline or not hasattr(pipeline, 'steps'):
            return cls(step, pipeline, processing_context)
            
        try:
            step_index = pipeline.steps.index(step)
            prev_step = pipeline.steps[step_index - 1] if step_index > 0 else None
            next_step = pipeline.steps[step_index + 1] if step_index < len(pipeline.steps) - 1 else None
            return cls(step, pipeline, processing_context, prev_step, next_step)
        except ValueError:
            # Step not found in pipeline
            return cls(step, pipeline, processing_context)


class FirstStepDiskInputRule(MaterializationRule):
    """Rule for inferring requires_fs_input for the first step in a pipeline."""
    
    def applies_to(self, context: RuleContext) -> bool:
        """
        This rule applies to the first step in a pipeline.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            True if this rule applies, False otherwise
        """
        return context.prev_step is None and context.pipeline is not None
        
    def infer_flags(self, context: RuleContext) -> Dict[str, bool]:
        """
        Infer requires_fs_input=True if input comes from disk.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            Dictionary of inferred flags
        """
        flags = {'inferred_requires_fs_input': False, 'inferred_requires_fs_output': False}
        
        if context.processing_context and hasattr(context.processing_context, 'get_step_input_dir'):
            input_dir = context.processing_context.get_step_input_dir(context.step)
            if input_dir and is_disk_path(input_dir):
                flags['inferred_requires_fs_input'] = True
                logger.debug("Rule %s inferred requires_fs_input=True for step %s",
                            self.name, context.step.__class__.__name__)
                
        return flags


class LastStepDiskOutputRule(MaterializationRule):
    """Rule for inferring requires_fs_output for the last step in a pipeline."""
    
    def applies_to(self, context: RuleContext) -> bool:
        """
        This rule applies to the last step in a pipeline.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            True if this rule applies, False otherwise
        """
        return context.next_step is None and context.pipeline is not None
        
    def infer_flags(self, context: RuleContext) -> Dict[str, bool]:
        """
        Infer requires_fs_output=True if output goes to disk.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            Dictionary of inferred flags
        """
        flags = {'inferred_requires_fs_input': False, 'inferred_requires_fs_output': False}
        
        if context.processing_context and hasattr(context.processing_context, 'get_step_output_dir'):
            output_dir = context.processing_context.get_step_output_dir(context.step)
            if output_dir and is_disk_path(output_dir):
                flags['inferred_requires_fs_output'] = True
                logger.debug("Rule %s inferred requires_fs_output=True for step %s",
                            self.name, context.step.__class__.__name__)
                
        return flags


class UpstreamRequiresFSInputRule(MaterializationRule):
    """Rule for inferring requires_fs_output based on next step's requirements."""
    
    def applies_to(self, context: RuleContext) -> bool:
        """
        This rule applies to any step with a next step.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            True if this rule applies, False otherwise
        """
        return context.next_step is not None
        
    def infer_flags(self, context: RuleContext) -> Dict[str, bool]:
        """
        Infer requires_fs_output=True if next step requires filesystem input.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            Dictionary of inferred flags
        """
        flags = {'inferred_requires_fs_input': False, 'inferred_requires_fs_output': False}
        
        # If N+1 has requires_fs_input=True, then N infers requires_fs_output=True
        if getattr(context.next_step, 'requires_fs_input', False):
            flags['inferred_requires_fs_output'] = True
            logger.debug("Rule %s inferred requires_fs_output=True for step %s because next step requires fs input",
                        self.name, context.step.__class__.__name__)
                
        return flags


class DownstreamRequiresFSOutputRule(MaterializationRule):
    """Rule for inferring requires_fs_input based on previous step's requirements."""
    
    def applies_to(self, context: RuleContext) -> bool:
        """
        This rule applies to any step with a previous step.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            True if this rule applies, False otherwise
        """
        return context.prev_step is not None
        
    def infer_flags(self, context: RuleContext) -> Dict[str, bool]:
        """
        Infer requires_fs_input=True if previous step requires filesystem output.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            Dictionary of inferred flags
        """
        flags = {'inferred_requires_fs_input': False, 'inferred_requires_fs_output': False}
        
        # If N has requires_fs_output=True, then N+1 infers requires_fs_input=True
        if getattr(context.prev_step, 'requires_fs_output', False):
            flags['inferred_requires_fs_input'] = True
            logger.debug("Rule %s inferred requires_fs_input=True for step %s because previous step requires fs output",
                        self.name, context.step.__class__.__name__)
                
        return flags


class MaterializationRuleRegistry:
    """Registry for materialization rules."""
    
    def __init__(self):
        """Initialize the rule registry."""
        self.rules = []
        
    def register_rule(self, rule: MaterializationRule) -> None:
        """
        Register a rule with the registry.
        
        Args:
            rule: The rule to register
        """
        self.rules.append(rule)
        
    def get_applicable_rules(self, context: RuleContext) -> List[MaterializationRule]:
        """
        Get all rules that apply to the given context.
        
        Args:
            context: The rule evaluation context
            
        Returns:
            List of applicable rules
        """
        return [rule for rule in self.rules if rule.applies_to(context)]
    
    def list_rules(self) -> List[MaterializationRule]:
        """
        Return all registered rule instances in definition order.
        
        Returns:
            List of registered rules
        """
        return self.rules


class FlagInferenceEngine:
    """Engine for inferring materialization flags."""
    
    def __init__(self, registry: Optional[MaterializationRuleRegistry] = None):
        """
        Initialize the flag inference engine.
        
        Args:
            registry: The rule registry to use (optional)
        """
        self.registry = registry or MaterializationRuleRegistry()
        
        # Register default rules if registry is empty
        if not self.registry.rules:
            self._register_default_rules()
            
    def _register_default_rules(self) -> None:
        """Register the default rules."""
        self.registry.register_rule(FirstStepDiskInputRule())
        self.registry.register_rule(LastStepDiskOutputRule())
        self.registry.register_rule(UpstreamRequiresFSInputRule())
        self.registry.register_rule(DownstreamRequiresFSOutputRule())
        
    def infer_flags(
        self, 
        step: Any, 
        pipeline: Optional[Any] = None, 
        context: Optional[Any] = None
    ) -> Dict[str, bool]:
        """
        Infer materialization flags for a step.
        
        Args:
            step: The step to check
            pipeline: The pipeline containing the step
            context: The processing context
            
        Returns:
            Dictionary of inferred flags
        """
        # Create rule context
        rule_context = RuleContext.from_pipeline_position(step, pipeline, context)
        
        # Get applicable rules
        applicable_rules = self.registry.get_applicable_rules(rule_context)
        
        # Apply rules and merge results
        inferred_flags = {
            'inferred_requires_fs_input': False,
            'inferred_requires_fs_output': False
        }
        
        for rule in applicable_rules:
            rule_flags = rule.infer_flags(rule_context)
            for key, value in rule_flags.items():
                if value:  # Only override False with True
                    inferred_flags[key] = True
                    
        return inferred_flags
