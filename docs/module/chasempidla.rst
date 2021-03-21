ChaseMpiDLA
---------------------

The class ``ChaseMpiDLAInterface`` defines the virtual functions required for the implementation
of ``ChaseMpiDLA``.

.. toctree::
   :maxdepth: 2

   chasempidlainterface



We provides multiple implementations of ``DLA`` targeting different computing architectures (CPUs or GPUs, sequential or parallel, shared-memory or distributed-memory) for ChASE.

.. toctree::
   :maxdepth: 3

   chasempidlaseq 
   chasempidlablaslapack
   chasempidlamultigpu


.. note::
    For the usage of these classes, please refer to :ref:`hello-world-chase`
    in the User Documentation.    