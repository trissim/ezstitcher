Advanced Configuration
======================

This example demonstrates advanced configuration options in EZStitcher.

Custom Configuration Files
--------------------------

You can create and save custom configurations:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig
    import json

    # Create a custom configuration
    config = PipelineConfig(
        reference_channels=["1", "2"],
        well_filter=["A01", "A02"],
        stitcher=StitcherConfig(
            tile_overlap=15.0,
            max_shift=75,
            margin_ratio=0.15
        ),
        focus_config=FocusAnalyzerConfig(
            method="laplacian",
            roi=(100, 100, 200, 200)
        ),
        reference_flatten="max_projection",
        stitch_flatten="best_focus",
        additional_projections=["max", "mean"]
    )

    # Save to JSON
    with open("my_config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Load from JSON
    with open("my_config.json", "r") as f:
        config_dict = json.load(f)
        loaded_config = PipelineConfig(**config_dict)

Configuration Inheritance
-------------------------

You can create derived configurations that inherit from a base configuration:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig

    # Create a base configuration
    base_config = PipelineConfig(
        reference_channels=["1"],
        stitcher=StitcherConfig(
            tile_overlap=10.0,
            max_shift=50
        )
    )

    # Create a derived configuration
    derived_config = PipelineConfig(
        **base_config.__dict__,  # Inherit all base config properties
        reference_channels=["1", "2"],  # Override reference channels
        well_filter=["A01", "A02"]      # Add well filter
    )

    # Create and run pipeline with derived configuration
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    pipeline = PipelineOrchestrator(derived_config)
    pipeline.run("path/to/plate_folder")

Dynamic Configuration
---------------------

You can dynamically create configurations based on image properties:

.. code-block:: python

    import numpy as np
    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    from ezstitcher.core.file_system_manager import FileSystemManager
    from pathlib import Path

    def create_dynamic_config(plate_folder):
        """Create a dynamic configuration based on image properties."""
        # Find a sample image
        fs_manager = FileSystemManager()
        sample_files = fs_manager.list_image_files(Path(plate_folder))
        if not sample_files:
            return PipelineConfig(reference_channels=["1"])
            
        sample_image = fs_manager.load_image(sample_files[0])
        
        # Analyze image properties
        mean_intensity = np.mean(sample_image)
        std_intensity = np.std(sample_image)
        
        # Determine if it's a Z-stack
        has_zstack, _ = fs_manager.detect_zstack_folders(plate_folder)
        
        # Create base configuration
        config = PipelineConfig(
            reference_channels=["1"]
        )
        
        # Adjust configuration based on image properties
        if has_zstack:
            config.reference_flatten = "max_projection"
            config.stitch_flatten = "best_focus"
            
            # Select focus method based on contrast
            if std_intensity / mean_intensity < 0.1:
                config.focus_config = FocusAnalyzerConfig(method="fft")
            else:
                config.focus_config = FocusAnalyzerConfig(method="combined")
        
        # Adjust stitcher configuration based on image size
        if sample_image.shape[0] > 2000:  # High-resolution image
            config.stitcher = StitcherConfig(
                tile_overlap=10.0,
                max_shift=100,  # Larger max_shift for high-res images
                margin_ratio=0.15
            )
        
        return config

    # Create dynamic configuration
    config = create_dynamic_config("path/to/plate_folder")

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Configuration Presets
---------------------

You can create configuration presets for common use cases:

.. code-block:: python

    from ezstitcher.core.config import PipelineConfig, StitcherConfig, FocusAnalyzerConfig

    def get_config_preset(preset_name):
        """Get a configuration preset by name."""
        if preset_name == "basic":
            return PipelineConfig(
                reference_channels=["1"],
                stitcher=StitcherConfig(
                    tile_overlap=10.0,
                    max_shift=50
                )
            )
        elif preset_name == "high_resolution":
            return PipelineConfig(
                reference_channels=["1"],
                stitcher=StitcherConfig(
                    tile_overlap=10.0,
                    max_shift=100,
                    margin_ratio=0.15
                )
            )
        elif preset_name == "z_stack_max":
            return PipelineConfig(
                reference_channels=["1"],
                reference_flatten="max_projection",
                stitch_flatten="max_projection"
            )
        elif preset_name == "z_stack_best_focus":
            return PipelineConfig(
                reference_channels=["1"],
                reference_flatten="max_projection",
                stitch_flatten="best_focus",
                focus_config=FocusAnalyzerConfig(
                    method="combined"
                )
            )
        elif preset_name == "multi_channel":
            return PipelineConfig(
                reference_channels=["1", "2"],
                reference_composite_weights={
                    "1": 0.7,
                    "2": 0.3
                }
            )
        else:
            return PipelineConfig(reference_channels=["1"])

    # Get a preset configuration
    config = get_config_preset("z_stack_best_focus")

    # Create and run pipeline
    from ezstitcher.core.processing_pipeline import PipelineOrchestrator
    pipeline = PipelineOrchestrator(config)
    pipeline.run("path/to/plate_folder")

Command Line Configuration
--------------------------

You can use the command line to specify configuration options:

.. code-block:: bash

    # Basic configuration
    ezstitcher /path/to/plate_folder --reference-channels 1 --tile-overlap 10

    # Z-stack configuration
    ezstitcher /path/to/plate_folder --reference-channels 1 --reference-flatten max --stitch-flatten best_focus

    # Well filtering
    ezstitcher /path/to/plate_folder --reference-channels 1 --wells A01 A02 B01 B02

    # Focus configuration
    ezstitcher /path/to/plate_folder --reference-channels 1 --focus-method combined --focus-roi 100 100 200 200

    # Multiple projections
    ezstitcher /path/to/plate_folder --reference-channels 1 --additional-projections max,mean
