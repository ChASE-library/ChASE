************************************
Installation and Setup on a Cluster
************************************

This section provides guidelines to install and setup the ChASE
library on computing clusters.
Each guideline is given as a step by step set of instructions to install ChASE on a
cluster either with or w/o support for GPUs. After the setup of ChASE,
users are provided with a description of how 
to link and integrate ChASE into their own codes. Such integration can
be achieved either by using one of the 
interfaces provided by the library, or by following our instructions
to implement the user's own interface.


Library Dependencies
====================

This section gives a brief introduction to the ChASE library dependencies and
on how to load the required libraries on a given supercomputer.

.. toctree::
   :maxdepth: 2

   installation/dependencies

Installation on Cluster
=========================

This section has two main goals: First, it provides the instructions
for the installation of ChASE on a given supercomputer
with or w/o multi-GPUs supports. Second, it describe how the user can
take advantage of a number of ready-to-use examples to build a 
simple driver and have a first try running ChASE on a cluster.


.. toctree::
   :maxdepth: 2

   installation/installcluster

Linking ChASE
=============

In order to embed the ChASE library in an application software, ChASE
can be opportunely linked following the instructions in this section.

.. toctree::
   :maxdepth: 2

   installation/link

Recommendation on the usage of Computing Resources
====================================================

Attaining the best performance with the available computing resources
requires to understand the inner working of the ChASE library. Since
the standard user is not expected to have such an understanding, this section
supplies a number of simple recommendations for the submission and
execution of jobs involving ChASE on a given computing cluster.

.. toctree::
   :maxdepth: 2

   installation/suggestion



