Annotations
===========

Manifests and other annotation files are built from the `SeSAMe package <https://zwdzwd.github.io/InfiniumAnnotation>`_ and illumina (cf `illumina docs <https://support.illumina.com.cn/downloads/infinium-methylationepic-v2-0-product-files.html>`_)

They are stored and versioned in the `pylluminator-data GitHub repository <https://github.com/eliopato/pylluminator-data/raw/main/>`_

Manifest
--------

Description of the columns of the `probe_infos.csv` file. If you want to use a custom manifest, you will need to provide this information.

``illumina_id`` : ID that matches probe IDs in .idat files

``probe_id`` : probe ID used in annotation files :

  * First letters : Either ``cg`` (CpG), ``ch`` (CpH), ``mu`` (multi-unique), ``rp`` (repetitive element), ``rs`` (SNP probes), ``ctl`` (control), ``nb`` (somatic mutations found in cancer)
  * Last 4 characters : top or bottom strand (``T/B``), converted or opposite strand (``C/O``), Infinium probe type (``1/2``), and the number of synthesis for representation of the probe on the array (``1,2,3,…,n``).

``type`` : probe type, Infinium-I or Infinium-II

``probe_type`` : ``cg`` (CpG), ``ch`` (CpH), ``mu`` (multi-unique), ``rp`` (repetitive element), ``rs`` (SNP probes), ``ctl`` (control), ``nb`` (somatic mutations found in cancer)

``channel``: color channel, green (methylated) or red (unmethylated)

``address_[A/B]``: Chip/tango address for A-allele and B-allele. For Infinium type I, allele A is Unmethylated, allele B is Methylated. For type II, address B is not set as there is only one probe. Addresses match the Illumina IDs found in IDat files.

``start``: the start position of the probe sequence

``end``: the end position of the probe sequence. Usually the start position +1 because probes typically span a single CpG site.

``chromosome``: chromosome number/letter

``mask_info``: name of the masks for this probe. Multiple masks are separated by semicolons. (details below)

``genes``: genes encoded by this sequence. Multiple gene names are separated by semicolons.

``transcript_types``: The types of transcripts linked to the probe's genomic location. These indicate whether the region corresponds to protein_coding, nonsense_mediated_decay, retained_intron, or other annotations. Multiple transcript types are separated by semicolons.

Masks
-----

Common masks
~~~~~~~~~~~~~

``M_mapping``: unmapped probes, or probes having too low mapping quality (alignment score under 35, either probe for Infinium-I) or Infinium-I probe allele A and B mapped to different locations

``M_nonuniq``: mapped probes but with mapping quality smaller than 10, either probe for Infinium-I

``M_uncorr_titration``: CpGs with titration correlation under 0.9. Functioning probes should have very high correlation with titrated methylation fraction.

Human masks (general and population-specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``M_commonSNP5_5pt``: mapped probes having at least a common SNP with MAF>=5% within 5bp from 3'-extension

``M_commonSNP5_1pt``: mapped probes having at least a common SNP with MAF>=1% within 5bp from 3'-extension

``M_1baseSwitchSNPcommon_1pt``: mapped Infinium-I probes with SNP (MAF>=1%) hitting the extension base and changing the color channel

``M_2extBase_SNPcommon_1pt``: mapped Infinium-II probes with SNP (MAF>=1%) hitting the extension base.

``M_SNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).

``M_1baseSwitchSNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).

``M_2extBase_SNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).

... more populations, e.g., ``EAS``, ``EUR``, ``AFR``, ``AMR``, ``SAS``.

Mouse masks (general and strain-specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``M_PWK_PhJ``: mapped probes having at least a PWK_PhJ strain-specific SNP within 5bp from 3'-extension

``M_1baseSwitchPWK_PhJ``: mapped Infinium-I probes with PWK_PhJ strain-specific SNP hitting the extension base and changing the color channel

``M_2extBase_PWK_PhJ``: mapped Infinium-II probes with PWK_PhJ strain-specific SNP hitting the extension base.

... more strains, e.g., ``AKR_J``, ``A_J``, ``NOD_ShiLtJ``, ``MOLF_EiJ``, ``129P2_OlaHsd`` ...

Genome information
------------------

``genome_info/gap_info.csv``: contains information on gaps in the genomic sequence. These gaps represent regions
that are not sequenced or that are known to be problematic in the data, such as areas that may have low coverage or difficult-to-sequence regions.

``genome_info/seq_length.csv``: keys are chromosome identifiers (e.g., 1, 2, ... X, etc.), and values are the corresponding sequence lengths (in base pairs).

``genome_info/transcripts_list.csv``: high-level overview of the transcripts and their boundaries (start and end positions).

``genome_info/transcripts_exons.csv``: information at the level of individual exons within each transcript (type, gene name, gene id...).
Details on `transcript_types` values can be found in `GRCh37 database <https://grch37.ensembl.org/info/genome/genebuild/biotypes.html>`_

``genome_info/chromosome_regions.csv``: Names, addresses and Giemsa stain pattern of all chromosomes' regions.

