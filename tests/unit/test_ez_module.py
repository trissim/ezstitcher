"""
Unit tests for the EZ module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ezstitcher.ez.core import EZStitcher
from ezstitcher.ez.functions import stitch_plate


class TestEZStitcher(unittest.TestCase):
    """Tests for the EZStitcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock for PipelineOrchestrator
        self.mock_orchestrator_patcher = patch('ezstitcher.ez.core.PipelineOrchestrator')
        self.mock_orchestrator_class = self.mock_orchestrator_patcher.start()
        self.mock_orchestrator = MagicMock()
        self.mock_orchestrator_class.return_value = self.mock_orchestrator
        self.mock_orchestrator.workspace_path = Path('/mock/workspace')
        
        # Create mock for AutoPipelineFactory
        self.mock_factory_patcher = patch('ezstitcher.ez.core.AutoPipelineFactory')
        self.mock_factory_class = self.mock_factory_patcher.start()
        self.mock_factory = MagicMock()
        self.mock_factory_class.return_value = self.mock_factory
        
        # Create test input path
        self.test_input = Path('/test/input')
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_orchestrator_patcher.stop()
        self.mock_factory_patcher.stop()
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        # Patch auto-detection methods
        with patch.object(EZStitcher, '_detect_z_stacks', return_value=True), \
             patch.object(EZStitcher, '_detect_channels', return_value=None):
            
            # Create EZStitcher instance
            stitcher = EZStitcher(self.test_input)
            
            # Check attributes
            self.assertEqual(stitcher.input_path, self.test_input)
            self.assertEqual(stitcher.output_path, self.test_input.parent / f"{self.test_input.name}_stitched")
            self.assertTrue(stitcher.normalize)
            self.assertTrue(stitcher.flatten_z)
            self.assertEqual(stitcher.z_method, "max")
            self.assertIsNone(stitcher.channel_weights)
            self.assertIsNone(stitcher.well_filter)
            
            # Check factory creation
            self.mock_factory_class.assert_called_once_with(
                input_dir=self.mock_orchestrator.workspace_path,
                output_dir=stitcher.output_path,
                normalize=True,
                flatten_z=True,
                z_method="max",
                channel_weights=None,
                well_filter=None
            )
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        # Create EZStitcher instance with custom parameters
        output_path = Path('/test/output')
        channel_weights = [0.7, 0.3]
        well_filter = ['A01', 'B02']
        
        stitcher = EZStitcher(
            self.test_input,
            output_path=output_path,
            normalize=False,
            flatten_z=False,
            z_method="focus",
            channel_weights=channel_weights,
            well_filter=well_filter
        )
        
        # Check attributes
        self.assertEqual(stitcher.input_path, self.test_input)
        self.assertEqual(stitcher.output_path, output_path)
        self.assertFalse(stitcher.normalize)
        self.assertFalse(stitcher.flatten_z)
        self.assertEqual(stitcher.z_method, "focus")
        self.assertEqual(stitcher.channel_weights, channel_weights)
        self.assertEqual(stitcher.well_filter, well_filter)
        
        # Check factory creation
        self.mock_factory_class.assert_called_once_with(
            input_dir=self.mock_orchestrator.workspace_path,
            output_dir=output_path,
            normalize=False,
            flatten_z=False,
            z_method="focus",
            channel_weights=channel_weights,
            well_filter=well_filter
        )
    
    def test_set_options(self):
        """Test setting options."""
        # Patch auto-detection methods
        with patch.object(EZStitcher, '_detect_z_stacks', return_value=True), \
             patch.object(EZStitcher, '_detect_channels', return_value=None):
            
            # Create EZStitcher instance
            stitcher = EZStitcher(self.test_input)
            
            # Reset mock to clear initialization call
            self.mock_factory_class.reset_mock()
            
            # Set options
            stitcher.set_options(
                normalize=False,
                z_method="focus",
                channel_weights=[0.5, 0.5]
            )
            
            # Check attributes
            self.assertFalse(stitcher.normalize)
            self.assertEqual(stitcher.z_method, "focus")
            self.assertEqual(stitcher.channel_weights, [0.5, 0.5])
            
            # Check factory recreation
            self.mock_factory_class.assert_called_once_with(
                input_dir=self.mock_orchestrator.workspace_path,
                output_dir=stitcher.output_path,
                normalize=False,
                flatten_z=True,
                z_method="focus",
                channel_weights=[0.5, 0.5],
                well_filter=None
            )
    
    def test_stitch(self):
        """Test stitching process."""
        # Patch auto-detection methods
        with patch.object(EZStitcher, '_detect_z_stacks', return_value=True), \
             patch.object(EZStitcher, '_detect_channels', return_value=None):
            
            # Create EZStitcher instance
            stitcher = EZStitcher(self.test_input)
            
            # Create mock pipelines
            mock_pipelines = [MagicMock(), MagicMock()]
            self.mock_factory.create_pipelines.return_value = mock_pipelines
            
            # Run stitching
            result = stitcher.stitch()
            
            # Check pipeline creation and execution
            self.mock_factory.create_pipelines.assert_called_once()
            self.mock_orchestrator.run.assert_called_once_with(pipelines=mock_pipelines)
            
            # Check result
            self.assertEqual(result, stitcher.output_path)


class TestStitchPlate(unittest.TestCase):
    """Tests for the stitch_plate function."""
    
    def test_stitch_plate(self):
        """Test stitch_plate function."""
        # Create mock for EZStitcher
        mock_stitcher = MagicMock()
        mock_stitcher.stitch.return_value = Path('/test/output')
        
        # Patch EZStitcher class
        with patch('ezstitcher.ez.functions.EZStitcher', return_value=mock_stitcher) as mock_class:
            # Call stitch_plate
            input_path = Path('/test/input')
            output_path = Path('/test/output')
            result = stitch_plate(
                input_path,
                output_path=output_path,
                normalize=False,
                z_method="focus"
            )
            
            # Check EZStitcher creation
            mock_class.assert_called_once_with(
                input_path,
                output_path=output_path,
                normalize=False,
                z_method="focus"
            )
            
            # Check stitching
            mock_stitcher.stitch.assert_called_once()
            
            # Check result
            self.assertEqual(result, Path('/test/output'))
