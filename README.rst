|logo| Welcome to pylluminator
==============================

.. image:: https://img.shields.io/github/last-commit/eliopato/pylluminator.svg
   :target: https://github.com/eliopato/pylluminator/commits/dev
   :alt: Last commit

.. image:: https://img.shields.io/github/actions/workflow/status/eliopato/pylluminator/run_test.yml?branch=main
   :target: https://github.com/eliopato/pylluminator/actions
   :alt: Test Status

.. image:: https://img.shields.io/codecov/c/github/eliopato/pylluminator
   :target: https://codecov.io/gh/eliopato/pylluminator
   :alt: Code coverage

.. image:: https://readthedocs.org/projects/pylluminator/badge/?version=latest
   :target: https://pylluminator.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: ./LICENSE
   :alt: MIT License

`Tutorials <https://pylluminator.readthedocs.io/en/latest/tutorials.html>`_ | `API documentation <https://pylluminator.readthedocs.io/en/latest/api.html>`_ | `Source code <https://github.com/eliopato/pylluminator>`_ | `Release on pip <https://pypi.org/project/pylluminator/>`_

Pylluminator is a Python package designed to provide an efficient workflow for processing, analyzing, and visualizing DNA
methylation data. Pylluminator is inspired from the popular R packages `SeSAMe <https://bioconductor.org/packages/release/bioc/html/sesame.html>`_ and  `ChAMP <https://bioconductor.org/packages/release/bioc/html/ChAMP.html>`_.


Pylluminator supports the following Illumina's Infinium Beadchip array versions:

* human: 27k, 450k, MSA, EPIC, EPIC+, EPICv2
* mouse: MM285
* mammalian: Mammal40

.. |logo| image:: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/logo.png
    :width: 100px


Main functionalities
--------------------

* idat files parsing

* data preprocessing

  * Type-I probes channel inference
  * Dye bias correction (3 methods: using normalization control probes / linear scaling / non-linear scaling)
  * Detection p-value calculation (pOOBAH)
  * Background correction (NOOB)
  * Batch effect correction (ComBat)

* data analysis and visualisation

  * beta values (density, PCA, MDS, dendrogram...)
  * DMPs accounting for replicates / random effects, DMRs
  * CNV, CNS
  * pathway analysis with GSEApy (GSEA, ORA)

* quality control

Visualization examples:

.. list-table::

    * - .. figure:: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_1_-_Read_data_and_get_betas_16_0.png
            :target: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_1_-_Read_data_and_get_betas_16_0.png

            Fig 1. Samples beta values density

      - .. figure:: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_3_-_Calculate_DMP_and_DMR_15_0.png
            :target: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_3_-_Calculate_DMP_and_DMR_15_0.png

            Fig 2. Differentially methylated regions (DMRs)

    * - .. figure:: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_3_-_Calculate_DMP_and_DMR_17_1.png
            :target: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_3_-_Calculate_DMP_and_DMR_17_1.png

            Fig 3. Probes beta values associated with a specific gene

      - .. figure:: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_4_-_Copy_Number_Variation_9_0.png
            :target: https://raw.githubusercontent.com/eliopato/pylluminator/refs/heads/main/docs/images/tutorials_4_-_Copy_Number_Variation_9_0.png

            Fig 4. Copy number variations (CNVs)


Installation
------------

With pip
~~~~~~~~

You can install Pylluminator directly with:

.. code-block:: shell

    pip install pylluminator

From source
~~~~~~~~~~~

We recommend using a virtual environment with Python 3.13 or 3.12 to build pylluminator from source. Here is an example using Conda.

Setup the virtual environment (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't have Conda installed yet, here are the instructions depending on your OS : `Windows <https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html>`_ | `Linux <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_ | `MacOS <https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_.
After installing it, make sure you have Pip installed by running the following command in the terminal:

.. code-block:: shell

    conda install anaconda::pip

Now you can create a Conda environment named "pylluminator" and activate it. You can change the name to your liking ;)

.. code-block:: shell

    conda create -n pylluminator python=3.13
    conda activate pylluminator


Install pylluminator
^^^^^^^^^^^^^^^^^^^^^

You can download the latest source from github, or clone the repository with this command:

.. code-block:: shell

    git clone https://github.com/eliopato/pylluminator.git

Your are now ready to install the dependencies and the package :

.. code-block:: shell

    cd pylluminator
    pip install -r requirements.txt


Usage
-----

Refer to https://pylluminator.readthedocs.io/ for step-by-step tutorials and detailed documentation.

Contributing
------------
We welcome contributions! If you'd like to help improve the package, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and test them.
4. Submit a pull request describing your changes.

Bug reports / new features suggestion
-------------------------------------

If you encounter any bugs, have questions, or feel like the package is missing a very important feature, please open an issue on the `GitHub Issues <https://github.com/eliopato/pylluminator/issues>`_ page.

When opening an issue, please provide as much detail as possible, including:

- Steps to reproduce the issue
- The version of the package you are using
- Any relevant code snippets or error messages

License
-------

This project is licensed under the MIT License - see the `LICENSE <./LICENSE>`_ file for details.

Acknowledgements
----------------

This package is strongly inspired from `SeSAMe <https://bioconductor.org/packages/release/bioc/html/sesame.html>`_ and
includes code from `methylprep <https://github.com/FoxoTech/methylprep>`_ for .idat files parsing.

