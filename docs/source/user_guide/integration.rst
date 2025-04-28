=========================
Integration with Other Tools
=========================

EZStitcher can be integrated with other image processing and analysis tools to create comprehensive workflows.

Integration Examples
------------------

Integration with Deep Learning Frameworks
--------------------------------------

You can integrate EZStitcher with deep learning frameworks like TensorFlow or PyTorch:

.. code-block:: python

    import tensorflow as tf

    # Load a pre-trained model
    model = tf.keras.models.load_model('/path/to/model')

    def apply_deep_learning(images):
        """Apply deep learning model to images."""
        result = []
        for img in images:
            # Preprocess image for the model
            input_tensor = tf.convert_to_tensor(img[np.newaxis, ..., np.newaxis], dtype=tf.float32)

            # Run inference
            predictions = model.predict(input_tensor)

            # Post-process predictions
            segmentation_map = predictions[0, ..., 0]

            # Return the segmentation map
            result.append(segmentation_map)

        return result

    # Use in a pipeline
    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step

    # First create standard pipelines with AutoPipelineFactory
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()

    # Then add a custom pipeline for deep learning
    deep_learning_pipeline = Pipeline(
        steps=[
            # Apply deep learning model to stitched images
            Step(
                name="Deep Learning Segmentation",
                func=apply_deep_learning,
                input_dir=orchestrator.output_dir,  # Use stitched images from previous pipeline
                output_dir=Path(orchestrator.output_dir).parent / "segmented"
            )
        ],
        name="Deep Learning Pipeline"
    )

    # Add the deep learning pipeline to the list
    pipelines.append(deep_learning_pipeline)

Integration with Image Analysis Tools
----------------------------------

EZStitcher can be used as part of a larger workflow with other image analysis tools:

.. code-block:: python

    # Example integration with CellProfiler
    import subprocess
    import os

    def run_cellprofiler_analysis(input_dir, output_dir, pipeline_path):
        """Run CellProfiler analysis on processed images."""
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Run CellProfiler headless
        subprocess.run([
            "cellprofiler",
            "-c", "-r",
            "-p", pipeline_path,
            "-i", input_dir,
            "-o", output_dir
        ], check=True)

        return True

    # Use in a pipeline after stitching
    from ezstitcher.core import AutoPipelineFactory
    from ezstitcher.core.pipeline import Pipeline
    from ezstitcher.core.steps import Step

    # First create standard pipelines with AutoPipelineFactory
    factory = AutoPipelineFactory(
        input_dir=orchestrator.workspace_path,
        normalize=True
    )
    pipelines = factory.create_pipelines()

    # Then add a custom pipeline for CellProfiler analysis
    analysis_pipeline = Pipeline(
        steps=[
            # Run CellProfiler on stitched images
            Step(
                name="CellProfiler Analysis",
                func=lambda images: run_cellprofiler_analysis(
                    input_dir=orchestrator.output_dir,  # Use stitched images from previous pipeline
                    output_dir=Path(orchestrator.output_dir).parent / "analysis",
                    pipeline_path="/path/to/cellprofiler_pipeline.cppipe"
                ) and images,  # Return images unchanged
                input_dir=orchestrator.output_dir,
                output_dir=orchestrator.output_dir  # No need to change images
            )
        ],
        name="Analysis Pipeline"
    )

    # Add the analysis pipeline to the list
    pipelines.append(analysis_pipeline)

Next Steps
---------

Now that you understand how to integrate EZStitcher with other tools, you can:

* Create custom export functions for your specific analysis needs
* Integrate with your preferred deep learning framework
* Build comprehensive image analysis pipelines
* Automate end-to-end workflows from acquisition to analysis

For more advanced usage patterns, see the :doc:`advanced_usage` section.
