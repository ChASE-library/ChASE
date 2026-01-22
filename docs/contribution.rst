Contributing to ChASE
*********************

We welcome contributions to ChASE! This guide outlines the process for contributing to the project.

Getting Started
===============

1. Express Interest in Contributing
------------------------------------

If you're interested in contributing to ChASE, you have two options:

**Option A: Create an Issue or Pull Request on GitHub**

You can create an issue or pull request on the `GitHub repository <https://github.com/ChASE-library/ChASE>`_
to express your interest in contributing and describe what you'd like to work on. This
is just to let us know about your interest - you don't need to fork and contribute code
directly on GitHub.

.. note::
  Since GitHub serves as a mirror repository from JSC GitLab, you cannot directly fork
  and contribute on GitHub. The GitHub repository is automatically synchronized from
  GitLab, so all active development happens on GitLab. After you express your interest,
  we will contact you to proceed with the contribution process.

**Option B: Contact the ChASE Team Directly**

Alternatively, you can also contact us at `chase@fz-juelich.de <mailto:chase@fz-juelich.de>`_ with
a motivated request of collaboration. We will consider your request and get in touch
with you to evaluate if and how to give you access directly to the GitLab repository
where the major developments of this software is carried out.

2. Sign the Collaboration Agreement (CLA)
-------------------------------------------

If you have not contributed to the ChASE library before, we will ask you to agree to a
Collaboration Agreement (CLA) before your pull request can be approved.

Currently, there is no automatic mechanism to sign such an agreement. You need to:

* Download the file `CLA_ChASE.pdf <https://github.com/ChASE-library/ChASE/blob/master/CLA_ChASE.pdf>`_ (that is part of the repository)
* Print it
* Sign it
* Send it back to `chase@fz-juelich.de <mailto:chase@fz-juelich.de>`_

Upon reception of your signed CLA, your request will be reviewed and then
eventually go for the next step.

3. Get Access to JSC GitLab
----------------------------

After your request of contribution is approved and you've signed the CLA, the ChASE team will
invite you to create an account on JSC GitLab for contributing. Once you have access:

* Create an account on JSC GitLab (if you don't have one already)
* Accept the invitation to the ChASE repository
* Clone the repository from GitLab for development

All active development happens on GitLab, while GitHub serves as a mirror repository.

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

   [hellochase@jwlogin21 ChASE-format]$ git commit -m "test hooks"
   [clang-format] formatted and staged Impl/chase_gpu/chase_gpu.hpp
   [58-implement-distributed-memory-chase-extension-for-bse-matrix a1a7583] test hooks
    1 file changed, 6 insertions(+), 6 deletions(-)

