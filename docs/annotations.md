# Manifests

Manifests and other annotation files are downloaded from [zwdzwd's github page](https://zwdzwd.github.io/InfiniumAnnotation) to match R package "SeSAMe"

Manifest columns meaning (from zwzzwd's github docs and [illumina docs](https://support.illumina.com.cn/downloads/infinium-methylationepic-v2-0-product-files.html)) :
- `CpG_chrm, CpG_beg, CpG_end`: genomic coordinate for the target, length 2 for CpG, length 1 for SNP and CpH. beg is 0-based and end is 1-based like in bed files.
- `address_A, address_B`: **[required]** Chip/tango address for A-allele and B-allele. For Infinium type I, allele A is Unmethylated, allele B is Methylated. For type II, address B is not set as there is only one probe. Addresses match the Illumina IDs found in IDat files.
- `target`: CG if the probe measures CpG methylation, reference allele if otherwise
- `nextBase`: the actual extension base (on the probe strand). "R" (stands for G/A) for Infinium-II probes
- `channel`: **[required]** color channel, green (methylated) or red (unmethylated)
- `Probe_ID`:  **[required]** the probe ID. 
  - First letters : Either cg (CpG), ch (CpH), mu (multi-unique), rp (repetitive element), rs (SNP probes), ctl (control), nb (somatic mutations found in cancer)
  - Last 4 characters : top or bottom strand (T/B), converted or opposite strand (C/O), Infinium probe type (1/2), and the number of synthesis for representation of the probe on the array (1,2,3,…,n).
- `mapFlag_A, mapChrm_A, mapPos_A, mapQ_A, mapCigar_A, AlleleA_ProbeSeq, mapNM_A, mapAS_A, mapYD_A`:
Mapping information for allele A. NM is the number of mutations. AS represents alignment score. YD (f/r/n) represents the bisulfite strand. mapFlag can be used to determine direction of the probe sequence. 0 means upstream and 16 means downstream.
- `mapFlag_B, mapChrm_B, mapPos_B, mapQ_B, mapCigar_B, AlleleB_ProbeSeq, mapNM_B, mapAS_B, mapYD_B` :
Mapping information for allele B. Same as allele A.
- `type`: Infinium-I or Infinium-II