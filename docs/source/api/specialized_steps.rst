Specialized Steps
===============

.. module:: ezstitcher.core.specialized_steps

This module contains specialized step implementations that inherit from the regular Step class
and pre-configure parameters for common operations like Z-stack flattening, focus selection,
and channel compositing.

ZFlatStep
--------

.. py:class:: ZFlatStep(method="max", input_dir=None, output_dir=None, well_filter=None)

   Specialized step for Z-stack flattening.

   This step performs Z-stack flattening using the specified method.
   It pre-configures variable_components=['z_index'] and group_by=None.

   :param method: Projection method. Options: "max", "mean", "median", "min", "std", "sum"
   :type method: str
   :param input_dir: Input directory
   :type input_dir: str or Path, optional
   :param output_dir: Output directory
   :type output_dir: str or Path, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional

   Example usage:

   .. code-block:: python

      from ezstitcher.core.specialized_steps import ZFlatStep

      # Create a maximum intensity projection step
      step = ZFlatStep(
          method="max",
          input_dir=orchestrator.workspace_path
      )

      # Create a mean intensity projection step
      step = ZFlatStep(
          method="mean",
          input_dir=orchestrator.workspace_path
      )

FocusStep
--------

.. py:class:: FocusStep(focus_options=None, input_dir=None, output_dir=None, well_filter=None)

   Specialized step for focus-based Z-stack processing.

   This step finds the best focus plane in a Z-stack using FocusAnalyzer.
   It pre-configures variable_components=['z_index'] and group_by=None.

   :param focus_options: Dictionary of focus analyzer options:
                        - metric: Focus metric. Options: "combined", "normalized_variance",
                                 "laplacian", "tenengrad", "fft" or a dictionary of weights (default: "combined")
   :type focus_options: dict, optional
   :param input_dir: Input directory
   :type input_dir: str or Path, optional
   :param output_dir: Output directory
   :type output_dir: str or Path, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional

   Example usage:

   .. code-block:: python

      from ezstitcher.core.specialized_steps import FocusStep

      # Create a best focus step with default metric (combined)
      step = FocusStep(
          input_dir=orchestrator.workspace_path
      )

      # Create a best focus step with specific metric
      step = FocusStep(
          focus_options={'metric': 'laplacian'},
          input_dir=orchestrator.workspace_path
      )

      # Create a best focus step with custom weights
      step = FocusStep(
          focus_options={'metric': {'nvar': 0.4, 'lap': 0.4, 'ten': 0.1, 'fft': 0.1}},
          input_dir=orchestrator.workspace_path
      )

CompositeStep
-----------

.. py:class:: CompositeStep(weights=None, input_dir=None, output_dir=None, well_filter=None)

   Specialized step for creating composite images from multiple channels.

   This step creates composite images from multiple channels with specified weights.
   It pre-configures variable_components=['channel'] and group_by=None.

   :param weights: List of weights for each channel. If None, equal weights are used.
   :type weights: list, optional
   :param input_dir: Input directory
   :type input_dir: str or Path, optional
   :param output_dir: Output directory
   :type output_dir: str or Path, optional
   :param well_filter: Wells to process
   :type well_filter: list, optional

   Example usage:

   .. code-block:: python

      from ezstitcher.core.specialized_steps import CompositeStep

      # Create a composite step with equal weights
      step = CompositeStep(
          input_dir=orchestrator.workspace_path
      )

      # Create a composite step with custom weights (70% channel 1, 30% channel 2)
      step = CompositeStep(
          weights=[0.7, 0.3],
          input_dir=orchestrator.workspace_path
      )
