Welcome to illuminator
======================

Illuminator is a Python package designed to provide an efficient workflow for processing and analyzing DNA
methylation data, mirroring the functionalities of the popular R package `SeSAMe <https://bioconductor.org/packages/release/bioc/html/sesame.html>`_.

.. note::

   This project is under active development.


Main functionalities
--------------------

* idat files parsing

* data preprocessing

  * background correction
  * normalization
  * dye bias correction
  * pOOBAH

* data analysis and visualisation

  * beta values
  * DMP and DMR
  * CNV

* quality control


Installation
------------

We strongly advise to create a dedicated virtual environment of your choice to run illuminator. Here is the procedure to use Conda.

#. If you don't have Conda installed yet, here are the instructions depending on your OS : `Windows <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_ | `Linux <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ | `MacOS <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_.


#. After installing it, make sure you have Pip installed by running the following command in the terminal::

    conda install anaconda::pip

#. Now that you're all set, create a new environment with **Python 3.12**::

    # create a conda environment named "illuminator"; you can change the name to your liking
    conda create -n illuminator python=3.12
    conda activate illuminator

#. Time to install illuminator! ::

    pip install illuminator

That's it :)

**Troubleshooting**

If you encounter any issue, you might want to update conda, pip and setuptools ::

    conda update --all
    pip install --upgrade pip
    pip install --upgrade setuptools

