Manifests
=========

Manifests and other annotation files are downloaded from `zwdzwd's github page <https://zwdzwd.github.io/InfiniumAnnotation>`_ to match R package "SeSAMe"

Information compiled from zwzzwd's github docs and `illumina docs <https://support.illumina.com.cn/downloads/infinium-methylationepic-v2-0-product-files.html>`_

Manifest columns meaning
------------------------

* ``CpG_chrm, CpG_beg, CpG_end``: genomic coordinate for the target, length 2 for CpG, length 1 for SNP and CpH. beg is 0-based and end is 1-based like in bed files.
* ``target``: ``CG`` if the probe measures CpG methylation, reference allele if otherwise
* ``nextBase``: for infinium Type I probes, the actual extension base (on the probe strand) after bisulfite conversion (``A/C/T``)
* ``channel``: color channel, green (methylated) or red (unmethylated)
* ``type`` : probe type, Infinium-I or Infinium-II
* ``Probe_ID``: the probe ID.

  * First letters : Either ``cg`` (CpG), ``ch`` (CpH), ``mu`` (multi-unique), ``rp`` (repetitive element), ``rs`` (SNP probes), ``ctl`` (control), ``nb`` (somatic mutations found in cancer)
  * Last 4 characters : top or bottom strand (``T/B``), converted or opposite strand (``C/O``), Infinium probe type (``1/2``), and the number of synthesis for representation of the probe on the array (``1,2,3,…,n``).
* ``*_A``, ``*_B`` : Informations for alleles A and B

  * ``address_[A/B]``: Chip/tango address for A-allele and B-allele. For Infinium type I, allele A is Unmethylated, allele B is Methylated. For type II, address B is not set as there is only one probe. Addresses match the Illumina IDs found in IDat files.
  * ``mapFlag`` : can be used to determine direction of the probe sequence. 0 means upstream and 16 means downstream.
  * ``mapChrm``
  * ``mapPos``
  * ``mapQ``
  * ``mapCigar``
  * ``Allele[A/B]_ProbeSeq`` : sequence of the probe identified in ``Address[A/B]`` column
  * ``mapNM`` : number of mutations
  * ``mapAS`` : alignment score
  * ``mapYD`` : bisulfite strand (``f/r/n``)

Mask columns meaning
----------------------

* ``Probe_ID``: Probe ID
* ``mask``: ","-delimited description of masks
* ``maskUniq``: unique short form of the mask without additional info.
* ``M_general``: TRUE/FALSE, merged from other masks, for example :

  * human: ``M_mapping+M_nonuniq+M_SNPcommon_5pt+M_1baseSwitchSNPcommon_5pt+M_2extBase_SNPcommon_5pt``
  * mouse: ``M_mapping+M_nonuniq``

Common masks
~~~~~~~~~~~~~

* ``M_mapping``: unmapped probes, or probes having too low mapping quality (alignment score under 35, either probe for Infinium-I) or Infinium-I probe allele A and B mapped to different locations
* ``M_nonuniq``: mapped probes but with mapping quality smaller than 10, either probe for Infinium-I
* ``M_uncorr_titration``: CpGs with titration correlation under 0.9. Functioning probes should have very high correlation with titrated methylation fraction.

Human masks (general and population-specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``M_commonSNP5_5pt``: mapped probes having at least a common SNP with MAF>=5% within 5bp from 3'-extension
* ``M_commonSNP5_1pt``: mapped probes having at least a common SNP with MAF>=1% within 5bp from 3'-extension
* ``M_1baseSwitchSNPcommon_1pt``: mapped Infinium-I probes with SNP (MAF>=1%) hitting the extension base and changing the color channel
* ``M_2extBase_SNPcommon_1pt``: mapped Infinium-II probes with SNP (MAF>=1%) hitting the extension base.
* ``M_SNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).
* ``M_1baseSwitchSNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).
* ``M_2extBase_SNP_EAS_1pt``: EAS population-specific mask (MAF>=1%).
* ... more populations, e.g., ``EAS``, ``EUR``, ``AFR``, ``AMR``, ``SAS``.

Mouse masks (general and strain-specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``M_PWK_PhJ``: mapped probes having at least a PWK_PhJ strain-specific SNP within 5bp from 3'-extension
* ``M_1baseSwitchPWK_PhJ``: mapped Infinium-I probes with PWK_PhJ strain-specific SNP hitting the extension base and changing the color channel
* ``M_2extBase_PWK_PhJ``: mapped Infinium-II probes with PWK_PhJ strain-specific SNP hitting the extension base.
* ... more strains, e.g., ``AKR_J``, ``A_J``, ``NOD_ShiLtJ``, ``MOLF_EiJ``, ``129P2_OlaHsd`` ...

Gene annotations
----------------
From `GRCh37 database <https://grch37.ensembl.org/info/genome/genebuild/biotypes.html>`_

* ``IG (C/D/J/V) gene``: Immunoglobulin gene that undergoes somatic recombination
* ``TR (C/D/J/V) gene``: T cell receptor gene that undergoes somatic recombination

  * ``C``: Constant chain
  * ``D``: Diversity chain
  * ``J``: Joining chain
  * ``V``: Variable chain


* ``Nonsense Mediated Decay``: A transcript with a premature stop codon considered likely to be subjected to targeted degradation. Nonsense-Mediated Decay is predicted to be triggered where the in-frame termination codon is found more than 50bp upstream of the final splice junction.
* ``Processed transcript``: Gene/transcript that doesn't contain an open reading frame (ORF).

  * ``Long non-coding RNA (lncRNA)``: A non-coding gene/transcript >200bp in length

    * ``3' overlapping ncRNA``: Transcripts where ditag and/or published experimental data strongly supports the existence of long (>200bp) non-coding transcripts that overlap the 3'UTR of a protein-coding locus on the same strand.
    * ``Antisense``: Transcripts that overlap the genomic span (i.e. exon or introns) of a protein-coding locus on the opposite strand.
    * ``Macro lncRNA``: Unspliced lncRNAs that are several kb in size.
    * ``Non coding``: Transcripts which are known from the literature to not be protein coding.
    * ``Retained intron``: An alternatively spliced transcript believed to contain intronic sequence relative to other, coding, transcripts of the same gene.
    * ``Sense intronic``: A long non-coding transcript in introns of a coding gene that does not overlap any exons.
    * ``Sense overlapping``: A long non-coding transcript that contains a coding gene in its intron on the same strand.
    * ``lincRNA (long intergenic ncRNA)``: Transcripts that are long intergenic non-coding RNA locus with a length >200bp. Requires lack of coding potential and may not be conserved between species.

  * ``ncRNA``: A non-coding gene.

    * ``miRNA``: A small RNA (~22bp) that silences the expression of target mRNA.
    * ``miscRNA``: Miscellaneous RNA. A non-coding RNA that cannot be classified.
    * ``piRNA``: An RNA that interacts with piwi proteins involved in genetic silencing.
    * ``rRNA``: The RNA component of a ribosome.
    * ``siRNA``: A small RNA (20-25bp) that silences the expression of target mRNA through the RNAi pathway.
    * ``snRNA``: Small RNA molecules that are found in the cell nucleus and are involved in the processing of pre messenger RNAs
    * ``snoRNA``: Small RNA molecules that are found in the cell nucleolus and are involved in the post-transcriptional modification of other RNAs.
    * ``tRNA``: A transfer RNA, which acts as an adaptor molecule for translation of mRNA.
    * ``vaultRNA``: Short non coding RNA genes that form part of the vault ribonucleoprotein complex.

* ``Protein coding``: Gene/transcipt that contains an open reading frame (ORF).
* ``Protein coding CDS not defined``: Alternatively spliced transcript of a protein coding gene for which we cannot define a CDS.
* ``Protein coding LOF``: Not translated in the reference genome owing to a SNP/DIP but in other individuals/haplotypes/strains the transcript is translated. Replaces the polymorphic_pseudogene transcript biotype.
* ``Pseudogene``: A gene that has homology to known protein-coding genes but contain a frameshift and/or stop codon(s) which disrupts the ORF. Thought to have arisen through duplication followed by loss of function.

  * ``IG``: Inactivated immunoglobulin gene.
  * ``Polymorphic``: Pseudogene owing to a SNP/indel but in other individuals/haplotypes/strains the gene is translated.
  * ``Processed``: Pseudogene that lack introns and is thought to arise from reverse transcription of mRNA followed by reinsertion of DNA into the genome.
  * ``Transcribed``: Pseudogene where protein homology or genomic structure indicates a pseudogene, but the presence of locus-specific transcripts indicates expression. These can be classified into 'Processed', 'Unprocessed' and 'Unitary'.
  * ``Translated``: Pseudogenes that have mass spec data suggesting that they are also translated. These can be classified into 'Processed', 'Unprocessed'
  * ``Unitary``: A species specific unprocessed pseudogene without a parent gene, as it has an active orthologue in another species.
  * ``Unprocessed``: Pseudogene that can contain introns since produced by gene duplication.
* ``Readthrough``: A readthrough transcript has exons that overlap exons from transcripts belonging to two or more different loci (in addition to the locus to which the readthrough transcript itself belongs).
* ``Stop codon readthrough``: The coding sequence contains a stop codon that is translated (as supported by experimental evidence), and termination occurs instead at a canonical stop codon further downstream. It is currently unknown which codon is used to replace the translated stop codon, hence it is represented by 'X' in the protein sequence
* ``TEC (To be Experimentally Confirmed)``: Regions with EST clusters that have polyA features that could indicate the presence of protein coding genes. These require experimental validation, either by 5' RACE or RT-PCR to extend the transcripts, or by confirming expression of the putatively-encoded peptide with specific antibodies.