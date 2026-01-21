Contributing to ChASE
*********************

We welcome contributions to ChASE! This guide outlines the process for contributing to the project.

Getting Started
===============

1. Create an Issue on GitHub
-----------------------------

If you're interested in contributing to ChASE, please start by creating an issue on
GitHub expressing your interest and describing what you'd like to work on. This helps
us coordinate efforts and ensures your contribution aligns with the project's goals.

2. Set Up GitLab Account
-------------------------

ChASE is developed on JSC GitLab, while GitHub serves as a mirror repository. To
contribute code, you'll need to:

* Create an account on JSC GitLab
* Request access to the ChASE repository
* Clone the repository from GitLab for development

The GitHub repository is automatically synchronized from GitLab, so all active
development happens on GitLab.

Auto Formatting of Committed Codes
===================================

ChASE relies on ``clang-format`` to automatically format the codes pushed to the
repository. This ensures consistent code style across the codebase.

Prerequisites
-------------

Before committing code, make sure you have the following set up:

* **clang-format available**: On JSC machines, load the Clang module::

    module load Clang

* **Clone ChASE git repository**: Make sure you have cloned the ChASE repository
  from GitLab

* **Configure git hooks**: In the root directory of ChASE, configure git to use
  the project's hooks::

    git config core.hooksPath ./docs/hooks

Usage
-----

Once the prerequisites are met, you can proceed with your normal git workflow:

.. code-block:: bash

   git add {changed files}
   git commit -m "update xxxxx"

When you commit, the git hook will automatically format your code using ``clang-format``.
You'll see output like:

.. code-block:: text

   [wu7@jwlogin21 ChASE-format]$ git commit -m "test hooks"
   [clang-format] formatted and staged Impl/chase_gpu/chase_gpu.hpp
   [58-implement-distributed-memory-chase-extension-for-bse-matrix a1a7583] test hooks
    1 file changed, 6 insertions(+), 6 deletions(-)

