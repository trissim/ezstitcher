========
Overview
========

This section explains the core concepts of microscopy image stitching and how EZStitcher handles them.

EZStitcher is built around a flexible pipeline architecture that allows you to create custom image processing workflows. Understanding these core concepts will help you create effective image processing pipelines tailored to your specific needs.

Key Components
-------------

* **PipelineOrchestrator**: Coordinates the execution of pipelines across wells
* **Pipeline**: A sequence of processing steps
* **Step**: A single processing operation
* **Function Handling**: Patterns for processing different image components
* **Directory Structure**: How EZStitcher organizes files and directories

For a visual overview of the architecture, see :doc:`architecture_overview`.
