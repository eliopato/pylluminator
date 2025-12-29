Annotations
===========

Manifests and other annotation files are built from the `SeSAMe package <https://zwdzwd.github.io/InfiniumAnnotation>`_ and illumina (cf `illumina docs <https://support.illumina.com.cn/downloads/infinium-methylationepic-v2-0-product-files.html>`_) for all genome versions and array type.
An updated version of the manifest is also available for EPICv2/hg38, as defined by `<https://www.biorxiv.org/content/10.1101/2025.03.12.642895v2>`_

All the files, restructured to be used with pylluminator, are stored and versioned in the `pylluminator-data GitHub repository <https://github.com/eliopato/pylluminator-data/raw/main/>`_

Manifest (probe_infos)
----------------------

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

``promoter_or_body``: ``b`` for body, ``p`` or ``Promoter`` for promoter

``cgi``: position of the probe regarding the CpG island. Possible values: ``Island``, ``Shelf``, ``Shore``, ``OpenSea``


Masks
^^^^^

Common masks
"""""""""""""

``M_mapping``: unmapped probes, or probes having too low mapping quality (alignment score under 35, either probe for Infinium-I) or Infinium-I probe allele A and B mapped to different locations

``M_nonuniq``: mapped probes but with mapping quality smaller than 10, either probe for Infinium-I

``M_uncorr_titration``: CpGs with titration correlation under 0.9. Functioning probes should have very high correlation with titrated methylation fraction.

Human masks (general and population-specific)
"""""""""""""""""""""""""""""""""""""""""""""

``M_commonSNP5_5pt``: mapped probes having at least a common SNP with MAF>=5% within 5bp from 3'-extension

``M_commonSNP5_1pt``: mapped probes having at least a common SNP with MAF>=1% within 5bp from 3'-extension

``M_1baseSwitchSNPcommon_1pt``: mapped Infinium-I probes with SNP (MAF>=1%) hitting the extension base and changing the color channel

``M_2extBase_SNPcommon_1pt``: mapped Infinium-II probes with SNP (MAF>=1%) hitting the extension base.

``M_SNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).

``M_1baseSwitchSNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).

``M_2extBase_SNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).

... more populations, e.g., ``EAS``, ``EUR``, ``AFR``, ``AMR``, ``SAS``.

Mouse masks (general and strain-specific)
"""""""""""""""""""""""""""""""""""""""""

``M_PWK_PhJ``: mapped probes having at least a PWK_PhJ strain-specific SNP within 5bp from 3'-extension

``M_1baseSwitchPWK_PhJ``: mapped Infinium-I probes with PWK_PhJ strain-specific SNP hitting the extension base and changing the color channel

``M_2extBase_PWK_PhJ``: mapped Infinium-II probes with PWK_PhJ strain-specific SNP hitting the extension base.

... more strains, e.g., ``AKR_J``, ``A_J``, ``NOD_ShiLtJ``, ``MOLF_EiJ``, ``129P2_OlaHsd`` ...

Genome information
------------------

Gap info
^^^^^^^^

Contains information on gaps in the genomic sequence. These gaps represent regions that are not sequenced or that are known to be 
problematic in the data, such as areas that may have low coverage or difficult-to-sequence regions.

``chromosome``: number or name of the chromosome 

``start``: the start position of the gap

``end``: the end position of the gap 

``width``: the size of the gap

``strand``: strand of the gap, usually `*` (not specified)

``type``: region type. Possible values: `telomere`, `contig` (continuous region), `scaffold` (group of regions that might contain gaps), `heterochromatin` (tightly packed DNA, 
less transcriptionally active), `short_arm` (p arm of the chromosome) 

Sequence lengths
^^^^^^^^^^^^^^^^

Keys are chromosome identifiers (e.g., 1, 2, ... X, etc.), and values are the corresponding sequence lengths (in base pairs).

``chromosome``: number or name of the chromosome 

``seq_length``: chromosome size in number of base pairs

Transcript list 
^^^^^^^^^^^^^^^
Detail of the exons contained in each transcripts.

``group_name``: unique identifier for the transcript (e.g., ENST00000456328.2), corresponds to `transcript_id` in the `transcript_exons` file.

``start``: the start position of the exon

``end``: the end position of the exon

``width``: the size of the exon

``exon_number``: exon ID within the transcript

Transcript exons 
^^^^^^^^^^^^^^^^
Information at the level of groups of exons for each transcript (type, gene name, gene id...).
Details on `transcript_types` values can be found in `GRCh37 database <https://grch37.ensembl.org/info/genome/genebuild/biotypes.html>`_

``chromosome``: number or name of the chromosome 

``transcript_start``: start position of the transcript on the chromosome

``transcript_end``: end position of the transcript on the chromosome

``transcript_strand``: strand of the transcript, either '+' (forward) or '-' (reverse)

``transcript_id``: unique identifier for the transcript (e.g., ENST00000456328.2)

``transcript_type``: type of the transcript (e.g., processed_transcript, lncRNA, miRNA)

``transcript_name``: name of the transcript (e.g., DDX11L1-202, WASH7P-201)

``gene_name``: name of the gene associated with the transcript (e.g., DDX11L, WASH7P)

``gene_id``: unique identifier for the gene (e.g., ENSG00000223972.5)

``gene_type``: type of the gene (e.g., transcribed_unprocessed_pseudogene, protein_coding)

``source``: source of the annotation (e.g., HAVANA, ENSEMBL)

``level``: level of annotation confidence or quality, from 1 to 3

``cds_start``: start position of the coding sequence within the transcript, if the transcript is protein_coding

``cds_end``: end position of the coding sequence within the transcript, if the transcript is protein_coding

Chromosome regions 
^^^^^^^^^^^^^^^^^^

Names, addresses and Giemsa stain pattern of all chromosomes' regions.

``chromosome``: number or name of the chromosome

``start``: start position of the region on the chromosome

``end``: end position of the region on the chromosome

``name``: name of the region, e.g.`p36.33` where `p` means the region is on the short arm, or `q` for the long arm

``giemsa_staining``: Possible values: `gneg` for gene poor regions, `gpos25` for moderate gene density regions, `gpos50` for intermediate gene density regions, `gpos75` for high gene density regions, 
gpos100 for very high gene density regions, `gvar` for variable gene density (often polymorphic) regions, `acen` for the centromere, and `stalk` for the stalk