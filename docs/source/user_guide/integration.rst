=========================
Integration with Other Tools
=========================

EZStitcher can be integrated with other image processing and analysis tools to create comprehensive workflows.

Exporting Data for Analysis
-------------------------

After processing with EZStitcher, you can export data for analysis with other tools:

.. code-block:: python

    import numpy as np
    from skimage import io
    import pandas as pd

    def export_for_analysis(stitched_image_path, output_csv):
        """Export image data for analysis."""
        # Load the stitched image
        image = io.imread(stitched_image_path)

        # Extract features (example: mean intensity in regions)
        regions = []
        for i in range(0, image.shape[0], 100):
            for j in range(0, image.shape[1], 100):
                region = image[i:i+100, j:j+100]
                regions.append({
                    'x': j,
                    'y': i,
                    'mean_intensity': np.mean(region),
                    'std_intensity': np.std(region),
                    'min_intensity': np.min(region),
                    'max_intensity': np.max(region)
                })

        # Save as CSV for analysis
        df = pd.DataFrame(regions)
        df.to_csv(output_csv, index=False)

        return df

    # Use in a pipeline
    from ezstitcher.core.steps import Step

    # Create a pipeline with export step
    export_pipeline = Pipeline(
        steps=[
            # Process and stitch images
            # ...

            # Export data for analysis
            Step(
                name="Export Data",
                func=lambda images: export_for_analysis(
                    stitched_image_path=dirs['stitched'] / "A01_stitched.tif",
                    output_csv=dirs['stitched'] / "A01_analysis.csv"
                ) and images,  # Return images unchanged
                input_dir=dirs['stitched'],
                output_dir=dirs['stitched']
            )
        ],
        name="Export Pipeline"
    )

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
    deep_learning_pipeline = Pipeline(
        steps=[
            # Process images
            # ...

            # Apply deep learning model
            Step(
                name="Deep Learning Segmentation",
                func=apply_deep_learning,
                input_dir=dirs['processed'],
                output_dir=dirs['segmented']
            )
        ],
        name="Deep Learning Pipeline"
    )

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

    # Use in a step after processing
    analysis_step = Step(
        name="CellProfiler Analysis",
        func=lambda images: run_cellprofiler_analysis(
            input_dir=dirs['stitched'],
            output_dir=dirs['analysis'],
            pipeline_path="/path/to/cellprofiler_pipeline.cppipe"
        ) and images,  # Return images unchanged
        input_dir=dirs['stitched'],
        output_dir=dirs['stitched']  # No need to change images
    )

Next Steps
---------

Now that you understand how to integrate EZStitcher with other tools, you can:

* Create custom export functions for your specific analysis needs
* Integrate with your preferred deep learning framework
* Build comprehensive image analysis pipelines
* Automate end-to-end workflows from acquisition to analysis

For more advanced usage patterns, see the :doc:`advanced_usage` section.
