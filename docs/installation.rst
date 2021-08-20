************
Installation
************

Trying to install bioinformatics software can often lead to headaches, so I've dedicated a lot of time to making MiniContourFinder easy to install regardless of operating system. However, installing and using MiniContourFinder requires basic knowledge of the command line and either of the common package manager `pip <https://pip.pypa.io/en/stable/#>`_ or `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

###############
``pip`` install
###############

Currently, the easiest way to install MiniContourFinder is with ``pip``. Chances are that you already have ``pip`` installed because it comes prepackaged with Python (at least since ``v3.4``/``v2.7.9``). To check if ``pip`` is installed, open up a terminal and check if the following outputs look similar.

.. code-block::

  $ python --version
  Python 3.N.N
  $ python -m pip --version
  pip X.Y.Y from ... (python 3.N.N)


If your default version of Python is ``2.N.N`` instead of ``3.N.N``, try the same commands, but with ``python3`` instead of ``python``, and ``pip3`` instead of ``pip``. If that didn't work, follow the installation guide `here <https://pip.pypa.io/en/stable/installation/>`_.


If it did, you can now install MiniContourFinder simply by running

.. code-block:: bash

  $ pip install MiniContourFinder

It will produce an output detailing the requirements that were already satisfied and that were missing (and installed along with MiniContourFinder)

.. code-block:: bash

  Collecting MiniContourFinder
    Downloading MiniContourFinder-1.0.14-py3-none-any.whl (47 kB)
      |████████████████████████████████| 47 kB 3.9 MB/s
  ...
  Installing collected packages: MiniContourFinder...

  Successfully installed MiniContourFinder-1.0.14...

#############
Conda install
#############

I've also published MiniContourFinder on Anaconda, `here <https://anaconda.org/iangilman/minicontourfinder>`_, but the install isn't functional yet. In the future, conda will be the prefered installation platform. To check if ``conda`` is already installed, you can enter ``conda info`` in a terminal, which should print information about your installation.

If ``conda`` is not installed, you can install it easily through a download, `here <https://docs.anaconda.com/anaconda/install/>`_, or use the command line, which I prefer and outline below. ``conda`` comes in either Anaconda, or Miniconda; the difference between the two being that Anaconda comes with loads of things preinstalled, while Miniconda is more bare bones. I prefer Miniconder, and will show an example macOS install below, but you read more about the differences `here <https://docs.conda.io/projects/conda/en/latest/user-guide/index.html>`_. 

First, copy the repsective `64-bit install link <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_ from the conda docs for your operating system (note that for macOS you want the "bash", not "pkg" version). Open a terminal and run

.. code-block:: bash

  $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

*That's a capital "o", not a zero*. This will download the installer. Next, run the installed with

.. code-block:: bash

  $ bash Miniconda3-latest-MacOSX-x86_64.sh

After following all the prompts and reading the license you should have ``conda`` installed! The last thing to do is "restart" your terminal so that is recognizes ``conda``. This can be done by quiting and reopening your terminal, or with the following

.. code-block:: bash

  $ source ~/.bash_profile

Now when you type ``conda info`` in your terminal you should return the installation info.

##############
GitHub install
##############

MiniContourFinder can be installed by cloning the `GitHub repo <https://github.com/isgilman/MiniContourFinder>`_.

.. code-block:: bash

  $ git clone https://github.com/isgilman/MiniContourFinder
  $ cd MiniContourFinder
  $ python setup.py install

###########################
Verifying your installation
###########################

I haven't gotten around to writing install tests yet, but you can check to see that your install is working by typing

.. code-block:: bash

  $ mcf -h

which should bring up the help info.

############
Uninstalling
############

*************************
Uninstalling with ``pip``
*************************

Uninstalling with ``pip`` is just as easy. My version lives in a directory called ``junkdrawer``.

.. code-block:: bash

  $ pip uninstall MiniContourFinder
  Found existing installation: MiniContourFinder 1.0.14
  Uninstalling MiniContourFinder-1.0.14:
    Would remove:
      junkdrawer/bin/mcf
      junkdrawer/bin/mcf_gui
      junkdrawer/bin/mcf_parallel
      junkdrawer/lib/python3.8/site-packages/MCF/*
      junkdrawer/lib/python3.8/site-packages/MiniContourFinder-1.0.14.dist-info/*

Then enter ``y`` or ``Y`` at the prompt.

.. code-block:: bash

  Proceed (Y/n)? y
    Successfully uninstalled MiniContourFinder-1.0.14

***************************
Uninstalling with ``conda``
***************************

***************************
Uninstalling GitHub install
***************************

If you installed MiniContourFinder from the GitHub repo, you can uninstall it with ``pip``, as above.

.. code-block:: bash

  $ pip uninstall MiniContourFinder

After that finished, navigate to the directory you downloaded the repo to, and delete it. My installation lives in ``junkdrawer``.

.. code-block:: bash

  $ cd junkdrawer
  $ rm -r MiniContourFinder