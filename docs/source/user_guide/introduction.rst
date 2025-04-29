============
Introduction
============

What is EZStitcher?
------------------

EZStitcher is a Python library designed to simplify the processing and stitching of microscopy images. It provides a flexible pipeline architecture that allows researchers to easily process large microscopy datasets, create composite images, flatten Z-stacks, and stitch tiled images together.

Why EZStitcher?
--------------

Modern microscopy generates complex datasets with multiple channels, Z-stacks, and tiled images. While powerful stitching tools exist, they often require significant programming expertise or lack support for advanced workflows. EZStitcher bridges this gap by:

* Providing an intuitive interface for non-programmers
* Maintaining the flexibility needed by advanced users
* Automating common microscopy workflows
* Reducing processing errors through standardized pipelines
* Building on the robust Ashlar stitching engine

Key Features
-----------

* :doc:`../user_guide/basic_usage`: Process and stitch images with minimal code using the EZ module
* :doc:`../concepts/pipeline`: Organize processing steps in a logical sequence
* :doc:`../concepts/directory_structure`: Protect original data while maintaining organized outputs
* :doc:`../concepts/function_handling`: Apply various processing functions in different patterns
* :doc:`../user_guide/microscope_formats`: Work with data from different microscope types
* :doc:`../concepts/pipeline_orchestrator`: Process multiple wells in parallel for faster results

For a detailed understanding of how these components work together, see :doc:`../concepts/architecture_overview`.

Supported Microscope Types
------------------------

EZStitcher currently supports multiple microscope types, including ImageXpress and Opera Phenix. For detailed information about supported microscope types, including file formats, naming conventions, and directory structures, see :ref:`microscope-formats` and :ref:`microscope-comparison`.

Support for additional microscope types can be added by implementing the appropriate interfaces. See the :doc:`../development/extending` guide for details.

Next Steps
---------

* :doc:`../getting_started/getting_started` - Install EZStitcher and run your first pipeline
* :doc:`basic_usage` - Learn about the simplified interface for non-coders
* :doc:`intermediate_usage` - Create custom pipelines with steps
* :doc:`best_practices` - Learn best practices for using EZStitcher

EZStitcher is designed for researchers working with microscopy data who need a flexible, code-based approach to image processing and stitching.

Getting Started
---------------------

For installation instructions and a quick introduction with examples, see the :doc:`../getting_started/getting_started` guide.

Key Concepts
-----------

EZStitcher is built around several key concepts that work together to provide a flexible and powerful image processing framework. For detailed information about each concept, see the corresponding documentation:

* :doc:`../concepts/architecture_overview` - Overview of EZStitcher's architecture
* :doc:`../concepts/pipeline_orchestrator` - Details about the Orchestrator
* :doc:`../concepts/pipeline` - Details about Pipelines
* :doc:`../concepts/step` - Details about Steps
* :doc:`../concepts/function_handling` - Details about function handling
* :doc:`../concepts/directory_structure` - Details about directory structure
* :doc:`best_practices` - Best practices for using EZStitcher

For a comprehensive understanding of EZStitcher's architecture and concepts, please refer to the :doc:`../concepts/index` section.


