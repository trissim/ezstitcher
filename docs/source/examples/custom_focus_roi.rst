Custom Focus ROI
==============

This example demonstrates how to use a custom region of interest (ROI) for focus detection.

Basic Focus ROI
------------

.. code-block:: python

    from ezstitcher.core.config import FocusAnalyzerConfig, ZStackProcessorConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor

    # Create configuration with custom focus ROI
    focus_config = FocusAnalyzerConfig(
        method="combined",
        roi=(50, 50, 100, 100)  # (x, y, width, height)
    )

    # Create projections first to ensure reference directory exists
    zstack_config = ZStackProcessorConfig(
        create_projections=True,
        projection_types=["max"],
        focus_detect=True
    )

    plate_config = PlateProcessorConfig(
        reference_channels=["1"],
        focus_analyzer=focus_config,
        z_stack_processor=zstack_config
    )

    # Create and run the processor
    processor = PlateProcessor(plate_config)
    processor.run("path/to/plate_folder")

Multiple ROIs for Different Wells
------------------------------

.. code-block:: python

    from ezstitcher.core.config import FocusAnalyzerConfig, ZStackProcessorConfig, PlateProcessorConfig
    from ezstitcher.core.plate_processor import PlateProcessor

    # Define different ROIs for different wells
    roi_map = {
        "A01": (50, 50, 100, 100),   # ROI for well A01
        "A02": (75, 75, 150, 150),   # ROI for well A02
        "default": (0, 0, 200, 200)  # Default ROI for other wells
    }

    # Create a function to select ROI based on well
    def get_roi_for_well(well_id):
        return roi_map.get(well_id, roi_map["default"])

    # Create configuration with dynamic ROI selection
    focus_config = FocusAnalyzerConfig(
        method="combined",
        roi=None  # We'll set this dynamically
    )

    zstack_config = ZStackProcessorConfig(
        create_projections=True,
        projection_types=["max"],
        focus_detect=True
    )

    plate_config = PlateProcessorConfig(
        reference_channels=["1"],
        focus_analyzer=focus_config,
        z_stack_processor=zstack_config
    )

    # Create processor
    processor = PlateProcessor(plate_config)

    # Custom processing function that sets ROI based on well
    def process_well(well_id, well_folder):
        # Set ROI for this well
        processor.config.focus_analyzer.roi = get_roi_for_well(well_id)
        
        # Process the well
        return processor.process_well(well_id, well_folder)

    # Process each well with custom ROI
    processor.process_well_func = process_well
    processor.run("path/to/plate_folder")
