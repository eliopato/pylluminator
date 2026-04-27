Tool comparison
----------------

Several tools are available for Illumina microarray data analysis, either in R (e.g. `SeSAMe <https://bioconductor.org/packages/release/bioc/html/sesame.html>`_, `ChAMP <https://bioconductor.org/packages/release/bioc/html/ChAMP.html>`_) or in Python (e.g. `Mepylome <https://mepylome.readthedocs.io/en/latest/>`_, `CpGTools <https://cpgtools.readthedocs.io/en/latest/overview.html>`_, `Methylsuite <https://life-epigenetics-methylprep.readthedocs-hosted.com/en/latest/>`_). 
The following summary highlights the main features of each tool and can guide the selection of the most appropriate option for a given study and dataset.

Unless otherwise stated, Pylluminator provides at least the same functionality as SeSAMe and ChAMP for each method mentioned, since it builds upon these tools. This overview is not exhaustive, and was conducted on the latest available package versions as of April 2026

Compared versions:
   * Pylluminator 2.2
   * SeSAMe 1.28.1
   * ChAMP 3.22
   * Mepylome 0.10.1
   * Methylsuite 1.4.0 (not maintained since 2022)
   * CpGTools 2.0.4

.. role:: raw-html(raw)
   :format: html


===========================   ====================  ====================  ====================      ====================       ====================   ====================
Array versions                 Pylluminator          SeSAMe                ChAMP                     Mepylome                   CpGTools                Methylsuite
===========================   ====================  ====================  ====================      ====================       ====================   ====================
27k                           :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`       :raw-html:`&#x274C;`   :raw-html:`&#x2705;`
450k                          :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x2705;`      :raw-html:`&#x2705;`       :raw-html:`&#x2705;`   :raw-html:`&#x2705;`
MSA                           :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x2705;`       :raw-html:`&#x2705;`   :raw-html:`&#x274C;`
EPIC                          :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x2705;`      :raw-html:`&#x2705;`       :raw-html:`&#x2705;`   :raw-html:`&#x2705;`
EPIC+                         :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`       :raw-html:`&#x274C;`   :raw-html:`&#x274C;`
EPICv2                        :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x2705;`       :raw-html:`&#x2705;`   :raw-html:`&#x274C;`
Updated EPICv2  [#f0]_        :raw-html:`&#x2705;`  :raw-html:`&#x274C;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`       :raw-html:`&#x274C;`   :raw-html:`&#x274C;`
Mammal40                      :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`       :raw-html:`&#x274C;`   :raw-html:`&#x274C;`
MM285                         :raw-html:`&#x2705;`  :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`       :raw-html:`&#x274C;`   :raw-html:`&#x2705;`
===========================   ====================  ====================  ====================      ====================       ====================   ====================

.. rubric:: Footnotes

.. [#f0] see *Re-annotating the EPICv2 manifest with genes, intragenic features, and regulatory elements*, `(BioRxiv link) <https://www.biorxiv.org/content/10.1101/2025.03.12.642895v2>`_
 
==================================     =============================   ====================  ====================      =============================       ======================   ====================
Data processing                        Pylluminator                    SeSAMe                ChAMP                     Mepylome                            CpGTools                  Methylsuite
==================================     =============================   ====================  ====================      =============================       ======================   ====================
Read .idat files                        :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x2705;`      :raw-html:`&#x2705;`                :raw-html:`&#x274C;`     :raw-html:`&#x2705;`
Type-I probes channel inference         :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`                :raw-html:`&#x274C;`     :raw-html:`&#x2705;`
Dye bias correction [#f1]_              :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`                :raw-html:`&#x274C;`     :raw-html:`&#x2705;`
pOOBAH                                  :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`                :raw-html:`&#x274C;`     :raw-html:`&#x2705;`
Normalization   [#f9]_                  :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x2705;`      :raw-html:`&#x2705;`                :raw-html:`&#x274C;`     :raw-html:`&#x2705;`
Compute beta values                     :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x2705;`      :raw-html:`&#x2705;`                :raw-html:`&#x274C;`     :raw-html:`&#x2705;`
Impute missing beta values [#f10]_      :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x2705;`      :raw-html:`&#x274C;`                :raw-html:`&#x274C;`     :raw-html:`&#x274C;`
Batch correction (ComBat) [#f2]_        :raw-html:`&#x2705;`           :raw-html:`&#x274C;`  :raw-html:`&#x2705;`      :raw-html:`&#x274C;`                :raw-html:`&#x2705;`     :raw-html:`&#x274C;`
Lift over                               :raw-html:`&#x2705;`           :raw-html:`&#x2705;`  :raw-html:`&#x274C;`      :raw-html:`&#x274C;`                :raw-html:`&#x274C;`     :raw-html:`&#x274C;`
==================================     =============================   ====================  ====================      =============================       ======================   ====================

.. rubric:: Footnotes

.. [#f1] Pylluminator includes 3 methods: using normalization control probes / linear scaling / non-linear scaling. It does not include the "most balanced" method of SeSAMe. MethylSuite only implements the non-linear correction.
.. [#f2] Pylluminator computes batch correction on M-values only; ChAMP has the option to use the beta values and CpGTools uses beta values only. The reason is that beta values "often deviates from a Gaussian distribution, exhibiting skewness and over-dispersion" (`read more <https://doi.org/10.1093/nargab/lqaf062>`_). Using the beta values for batch correction also results in values out of the 0-1 range.
.. [#f9] Pylluminator, SeSAMe and MethylSuite implement NOOB for normalization. Mepylome offers 3 normalization methods: NOOB, SWAN and Illumina. ChAMP offers 4 methods: PBC,BMIQ, SWAN and FunctionalNormalize.
.. [#f10] SeSAMe implements 3 methods for beta imputation: imputation by genomic neighbor, imputation by matrix mean, and imputation based on available public data. Pylluminator only implements imputation based on public data, with the data provided by SeSAMe. ChAMP implements KNN imputation or using a provided dataset to pick default values from.

========================================     =============================   =============================  =============================      ====================       =============================   ====================
Data analysis & visualisation                Pylluminator                    SeSAMe                         ChAMP                              Mepylome                   CpGTools                        Methylsuite
========================================     =============================   =============================  =============================      ====================       =============================   ====================
Beta values density  [#f3]_                  :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x274C;`           :raw-html:`&#x2705;`
Beta values dimension reduction  [#f4]_      :raw-html:`&#x2705;`            :raw-html:`&#x274C;`           :raw-html:`&#x2705;`               :raw-html:`&#x2705;`        :raw-html:`&#x2705;`           :raw-html:`&#x274C;`
Beta values dendrogram                       :raw-html:`&#x2705;`            :raw-html:`&#x274C;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
Beta values imputation                       :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
Quality control                              :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x2705;`           :raw-html:`&#x2705;`
Gene visualisation                           :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
Genomic regions analysis                     :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x2705;`           :raw-html:`&#x274C;`
Data inference (sex, age, etc)  [#f8]_       :raw-html:`&#x274C;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x2705;`           :raw-html:`&#x2705;`
DMPs  [#f7]_                                 :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x2705;`           :raw-html:`&#x2705;`
DMRs   [#f5]_                                :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x274C;`           :raw-html:`&#x2705;`
CNV                                          :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x2705;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
CNS                                          :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x274C;`               :raw-html:`&#x2705;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
GSEA [#f6]_                                  :raw-html:`&#x2705;`            :raw-html:`&#x2705;`           :raw-html:`&#x2705;`               :raw-html:`&#x274C;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
Interactive plot interface                   :raw-html:`&#x274C;`            :raw-html:`&#x274C;`           :raw-html:`&#x2705;`               :raw-html:`&#x2705;`        :raw-html:`&#x274C;`           :raw-html:`&#x274C;`
========================================     =============================   =============================  =============================      ====================       =============================   ====================

.. rubric:: Footnotes

.. [#f3] Pylluminator focuses on plotting beta density by sample and sample metadata (e.g control vs patient), SeSaMe on density by probes design (type I vs type II)
.. [#f4] Pylluminator offers 14 different methods (e.g., MDS, PCA, NMF, UMAP, FA), ChAMP implements MDS, Mepylome UMAP, CpGTools t-SNE, PCA, and UMAP.
.. [#f7] Pylluminator adds the option to account for random effects
.. [#f5] ChAMP offers several methods (bumphunter, dmrcate, probelasso), where SeSAMe and Pylluminator implement one. Methylsuite implements bumphunter and comb-p.
.. [#f6] Pylluminator offers a wide range of options for GSEA analysis, thanks to the `GSEApy <https://gseapy.readthedocs.io/en/latest/>`_ package. SeSAMe provides basic GSEA functions through the `KYCG <https://bioconductor.org/packages/release/bioc/html/knowYourCG.html>`_ package, that provides an advanced framework for CpG-centered analysis.
.. [#f8] CpGTools and MethylSuite implement sex inference only
