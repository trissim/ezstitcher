Pipeline Factory Functions
=======================

.. module:: ezstitcher.core

.. autofunction:: create_basic_pipeline

.. autofunction:: create_multichannel_pipeline

.. autofunction:: create_zstack_pipeline

.. autofunction:: create_focus_pipeline

Example Usage
-----------

Basic Pipeline
^^^^^^^^^^^^

.. code-block:: python

    from ezstitcher.core import create_basic_pipeline
    
    pipeline = create_basic_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        normalize=True
    )

Multi-Channel Pipeline
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from ezstitcher.core import create_multichannel_pipeline
    
    pipeline = create_multichannel_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        weights=[0.7, 0.3],
        stitch_channels_separately=True
    )

Z-Stack Pipeline
^^^^^^^^^^^^^

.. code-block:: python

    from ezstitcher.core import create_zstack_pipeline
    
    pipeline = create_zstack_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        method="projection",
        method_options={'method': 'max'}
    )

Focus Pipeline
^^^^^^^^^^^

.. code-block:: python

    from ezstitcher.core import create_focus_pipeline
    
    pipeline = create_focus_pipeline(
        input_dir="path/to/images",
        output_dir="path/to/output",
        metric="variance_of_laplacian"
    )
