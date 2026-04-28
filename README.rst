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
  * Missing beta values imputation
  * Lift over annotation

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

With uv (recommended)
~~~~~~~~~~~~~~~~~~~~~

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager. If you don't have it yet, install it with:

.. code-block:: shell

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then install Pylluminator into a uv-managed project:

.. code-block:: shell

    uv add pylluminator

Or with the optional GSEA extras:

.. code-block:: shell

    uv add "pylluminator[gsea]"
    
With pip
~~~~~~~~

You can install Pylluminator directly with:

.. code-block:: shell

    pip install pylluminator

Or, if you want to use the GSEA functionalities, install the additional dependencies with:

.. code-block:: shell

    pip install pylluminator[gsea]


From source
~~~~~~~~~~~

We recommend using `uv <https://docs.astral.sh/uv/>`_ to build pylluminator from source. The project requires Python 3.12 or later.

**Install uv (if needed)**

.. code-block:: shell

    curl -LsSf https://astral.sh/uv/install.sh | sh

**Clone and install**

.. code-block:: shell

    git clone https://github.com/eliopato/pylluminator.git
    cd pylluminator
    uv sync

This creates a virtual environment and installs all dependencies automatically. To include optional extras:

.. code-block:: shell

    uv sync --extra gsea
    uv sync --extra dev
    uv sync --extra docs

Run scripts or tests within the project environment using ``uv run``:

.. code-block:: shell

    uv run pytest

Usage
-----

Refer to https://pylluminator.readthedocs.io/ for step-by-step tutorials and detailed documentation.

Citing
-------

Pylluminator is described in detail in: 
*Pylluminator: fast and scalable analysis of DNA methylation data in Python*, available on `BioRxiv <https://www.biorxiv.org/content/10.1101/2025.09.16.676547v1>`_

If you use this package in your research, please cite our work.

If you use the updated version of the EPICv2/hg38 annotations, please cite *Re-annotating the EPICv2 manifest with genes, intragenic features, and regulatory elements*, `(BioRxiv link) <https://www.biorxiv.org/content/10.1101/2025.03.12.642895v2>`_


Contributing
------------
We welcome contributions! If you'd like to help improve the package, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and test them.
4. Submit a pull request describing your changes.

The packages used for development (testing, packaging and building the documentation) can be installed with:

.. code-block:: shell

    uv sync --extra dev --extra docs

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

