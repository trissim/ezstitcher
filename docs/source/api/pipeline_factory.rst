Pipeline Factory
==============

.. module:: ezstitcher.core.pipeline_factories

This module contains the AutoPipelineFactory class that creates pre-configured pipelines
for all common workflows, leveraging specialized steps to reduce boilerplate code.

For comprehensive information about pipeline factories, including:

* Pipeline structure and behavior
* Parameter descriptions and defaults
* Examples for different use cases
* Customization options

See :ref:`pipeline-factory-concept` in the :doc:`../concepts/pipeline_factory` documentation.

AutoPipelineFactory
-----------------

.. autoclass:: AutoPipelineFactory
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: create_pipelines
