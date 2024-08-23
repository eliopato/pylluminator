Welcome to illuminator
======================

Installation
------------

We strongly advise to create a dedicated Conda environment to run illuminator. If you don't have Conda installed yet, here are the instructions depending on your OS : `Windows <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_ | `Linux <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ | `MacOS <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_.

After installing it, make sure you have Pip installed by running the following command in the terminal::

    conda install anaconda::pip


Now that you're all set, create a new environment with Python 3.12::

    # create a conda environment named "illuminator"; you can change the name to your liking
    conda create -n illuminator python=3.12
    conda activate illuminator


Now it's time to install illuminator! ::

    pip install illuminator

And that's it :)

**Troubleshooting**
If you enconter any issue, you might want to update conda, pip and setuptools ::

    conda update --all
    pip install --upgrade pip
    pip install --upgrade setuptools

Usage
------------
todo :)