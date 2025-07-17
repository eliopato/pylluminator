import os
import numpy as np
import pandas as pd
import shutil
from pylluminator.annotations import ArrayType, GenomeVersion
from pylluminator.samples import read_samples, from_sesame
from pylluminator.utils import download_from_geo

def test_download_from_geo(data_path):
    geo_ids = ['GSM7698438', 'GSM7698446', 'GSM7698462', 'GSM7698435', 'GSM7698443', 'GSM7698459']
    download_from_geo(geo_ids, data_path)

    expected_files = [
        "GSM7698435_PREC_500_1_Grn.idat.gz",
        "GSM7698446_LNCAP_500_2_Grn.idat.gz",
        "GSM7698435_PREC_500_1_Red.idat.gz",
        "GSM7698446_LNCAP_500_2_Red.idat.gz",
        "GSM7698438_LNCAP_500_1_Grn.idat.gz",
        "GSM7698459_PREC_500_3_Grn.idat.gz",
        "GSM7698438_LNCAP_500_1_Red.idat.gz",
        "GSM7698459_PREC_500_3_Red.idat.gz",
        "GSM7698443_PREC_500_2_Grn.idat.gz",
        "GSM7698462_LNCAP_500_3_Grn.idat.gz",
        "GSM7698443_PREC_500_2_Red.idat.gz",
        "GSM7698462_LNCAP_500_3_Red.idat.gz"
    ]

    for file_name in expected_files:
        file_path = f'{data_path}/{file_name}'
        assert os.path.exists(file_path), f'File {file_path} does not exist'

def wrapper_array_version(path, gsm_id, expected_array_type, expected_genome_type):
    shutil.rmtree(path, ignore_errors=True)
    download_from_geo(gsm_id, path)
    samples = read_samples(path, annotation=None)
    assert samples.annotation.array_type == expected_array_type
    assert samples.annotation.genome_version == expected_genome_type

    # test some specificities (eg no control probes)
    samples.controls()
    samples.mask_xy_probes()
    samples.mask_quality_probes()
    if expected_array_type == ArrayType.HUMAN_27K:
        samples.get_normalization_controls()
        samples.dye_bias_correction()
    shutil.rmtree(path)

def test_mm285(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM5587107', ArrayType.MOUSE_MM285, GenomeVersion.MM39)

def test_hm27(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM443819', ArrayType.HUMAN_27K, GenomeVersion.HG38)

def test_hm450(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM2796256', ArrayType.HUMAN_450K, GenomeVersion.HG38)

def test_msa(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM8217992', ArrayType.HUMAN_MSA, GenomeVersion.HG38)

def test_epic(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM2432891', ArrayType.HUMAN_EPIC, GenomeVersion.HG38)

def test_epic_plus(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM5099142', ArrayType.HUMAN_EPIC_PLUS, GenomeVersion.HG38)

def test_epic_v2(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM7698446', ArrayType.HUMAN_EPIC_V2, GenomeVersion.HG38)

def test_mammal40(data_path_tmp):
    wrapper_array_version(data_path_tmp, 'GSM5581045', ArrayType.MAMMAL_40, GenomeVersion.HG38)

def test_read_samples(data_path):
    min_beads = 0
    max_samples = 5
    my_samples = read_samples(data_path, annotation=None, min_beads=min_beads, keep_idat=True, max_samples=max_samples)

    assert my_samples.sample_sheet is not None
    assert my_samples.idata is not None
    assert my_samples.min_beads == 1
    assert len(my_samples.idata ) == max_samples
    assert my_samples.nb_samples == max_samples
    assert my_samples.annotation.array_type == ArrayType.HUMAN_EPIC_V2
    assert my_samples.masks.number_probes_masked(sample_label='PREC_500_3') == 52

    # Check that the samples are correctly loaded
    sample = my_samples['PREC_500_3']
    assert len(sample) == 937688  # vs 937690 in the original data bc of 2 staining control probes with NA channels
    assert sample is not None

    # Check that the samples are correctly loaded
    assert len(my_samples.type1()) == 128295
    assert len(my_samples.type1_green()) == 45685
    assert len(my_samples.type1_red()) == 82610
    assert len(my_samples.type2()) == 809393
    assert len(my_samples.type1()['PREC_500_3'].columns) == 4
    assert len(my_samples.type2()['PREC_500_3'].columns) == 2
    # test probe types
    assert len(my_samples.cg_probes()) == 933252
    assert len(my_samples.ch_probes()) == 2914
    assert len(my_samples.snp_probes()) == 65
    # test meth and unmeth subsets
    assert len(my_samples.unmeth()) == 937688
    assert len(my_samples.meth()) == 937688
    assert len(my_samples.unmeth()['PREC_500_3'].columns) == 2
    assert len(my_samples.meth()['PREC_500_3'].columns) == 2

    ##################################################################################"

    # test type 1 out of band
    oob_probes = my_samples.oob()['PREC_500_3']
    assert len(oob_probes) == 128295
    oob_r = oob_probes.xs('cg00001261_BC11', level='probe_id').values[0]
    assert len(oob_r) == 4
    assert oob_r[0] == 305
    assert oob_r[1] == 346
    assert np.isnan(oob_r[2])
    assert np.isnan(oob_r[3])
    oob_g = oob_probes.xs('rs1414097_BC11', level='probe_id').values[0]
    assert len(oob_g) == 4
    assert np.isnan(oob_g[0])
    assert np.isnan(oob_g[1])
    assert oob_g[2] == 277
    assert oob_g[3] == 241

    # out of band red
    assert len(my_samples.oob_red()) == 45685
    oob_r = my_samples.oob_red().xs('rs1414097_BC11', level='probe_id')['PREC_500_3'].values[0]
    assert len(oob_r) == 2
    assert oob_r[0] == 277
    assert oob_r[1] == 241

    # out of band green
    assert len(my_samples.oob_green()) == 82610
    oob_g = my_samples.oob_green().xs('cg00001261_BC11', level='probe_id')['PREC_500_3'].values[0]
    assert len(oob_g) == 2
    assert oob_g[0] == 305
    assert oob_g[1] == 346

    ##################################################################################"

    # test type 1 in band
    ib_probes = my_samples.ib()['PREC_500_3']
    assert len(ib_probes) == 128295
    ib_r = ib_probes.xs('cg00001261_BC11', level='probe_id').values[0]
    assert len(ib_r) == 4
    assert np.isnan(ib_r[0])
    assert np.isnan(ib_r[1])
    assert ib_r[2] == 5396
    assert ib_r[3] == 11840
    ib_g = ib_probes.xs('rs1414097_BC11', level='probe_id').values[0]
    assert len(ib_g) == 4
    assert ib_g[0] == 636
    assert ib_g[1] == 687
    assert np.isnan(ib_g[2])
    assert np.isnan(ib_g[3])
    # in band red
    assert len(my_samples.ib_red()) == 82610
    ib_r = my_samples.ib_red().xs('cg00001261_BC11', level='probe_id')['PREC_500_3'].values[0]
    assert len(ib_r) == 2
    assert ib_r[0] == 5396
    assert ib_r[1] == 11840
    # in band green
    assert len(my_samples.ib_green()) == 45685
    ib_g = my_samples.ib_green().xs('rs1414097_BC11', level='probe_id')['PREC_500_3'].values[0]
    assert len(ib_g) == 2
    assert ib_g[0] == 636
    assert ib_g[1] == 687

    ##################################################################################"

    # check values for a probe of each type (Type I green, type I red, type II)
    probe_tI_r = my_samples._signal_df.xs('cg00003555_BC11', level='probe_id')['PREC_500_3']
    assert probe_tI_r[('R', 'U')].values == 7861.0
    assert probe_tI_r[('R', 'M')].values == 209.0
    assert probe_tI_r[('G', 'U')].values == 294.0
    assert probe_tI_r[('G', 'M')].values == 104.0

    probe_tII = my_samples._signal_df.xs('cg00003622_BC21', level='probe_id')['PREC_500_3']
    assert probe_tII[('R', 'U')].values == 360.0
    assert np.isnan(probe_tII[('R', 'M')].values)
    assert np.isnan(probe_tII[('G', 'U')].values)
    assert probe_tII[('G', 'M')].values == 636.0

    probe_tII_g = my_samples._signal_df.xs('cg00003625_TC11', level='probe_id')['PREC_500_3']
    assert probe_tII_g[('R', 'U')].values == 445.0
    assert probe_tII_g[('R', 'M')].values == 319.0
    assert probe_tII_g[('G', 'U')].values == 2827.0
    assert probe_tII_g[('G', 'M')].values == 1522.0

def test_read_sample_wrong_input(data_path):
    assert read_samples(data_path, sample_sheet_df=pd.DataFrame(), sample_sheet_name='sample.csv') is None
    assert read_samples(data_path, sample_sheet_name='nonexistent.csv') is None

def test_from_sesame(test_samples, tmpdir):
    assert from_sesame('fakedir', test_samples.annotation) is None
    assert from_sesame(['liste'], test_samples.annotation) is None  # input cant be a list

    csv_content = ('"","Probe_ID","MG","MR","UG","UR","col","mask"\n'
     '"1","cg00000029_TC21",NA,NA,169,3222,"2",FALSE\n'
     '"2","cg00000109_TC21",NA,NA,1664,577,"2",FALSE\n'
     '"3","cg00000155_BC21",NA,NA,1901,404,"2",FALSE\n'
     '"4","cg00000158_BC21",NA,NA,2163,438,"2",FALSE\n'
     '"5","cg00000165_TC21",NA,NA,215,2370,"2",TRUE\n'
     '"6","cg00000221_BC21",NA,NA,1195,695,"2",FALSE\n'
     '"7","cg00000236_TC21",NA,NA,1491,433,"2",FALSE\n'
     '"8","cg00000289_TC21",NA,NA,406,289,"2",TRUE\n'
     '"9","cg00000292_BC21",NA,NA,3596,3219,"2",FALSE\n')

    file_path = f'{tmpdir}/sesame1.csv'
    with open(file_path, 'w') as f:
        f.write(csv_content)

    sample1 = from_sesame(file_path, test_samples.annotation)
    assert sample1 is not None
    assert sample1.sample_labels == ['sesame1']
    assert sample1.nb_probes == 9
    assert sample1.masks.number_probes_masked(sample_label='sesame1') == 2

    csv_content = ''',Probe_ID,MG,MR,UG,UR,col,mask\n
    1,cg00000029_TC21,NA,NA,187,4268,2,FALSE\n
    2,cg00000109_TC21,NA,NA,1823,885,2,FALSE\n
    3,cg00000155_BC21,NA,NA,2107,347,2,FALSE\n
    4,cg00000158_BC21,NA,NA,2841,436,2,FALSE\n
    5,cg00000165_TC21,NA,NA,197,2445,2,FALSE\n
    6,cg00000221_BC21,NA,NA,1375,658,2,FALSE\n
    7,cg00000236_TC21,NA,NA,1829,663,2,FALSE\n
    8,cg00000289_TC21,NA,NA,599,166,2,FALSE\n
    9,cg00000292_BC21,NA,NA,4161,3318,2,FALSE\n
    '''
    file_path = f'{tmpdir}/sesame2.csv'
    with open(file_path, 'w') as f:
        f.write(csv_content)

    samples = from_sesame(tmpdir, test_samples.annotation)
    assert samples is not None
    assert 'sesame1' in samples.sample_labels
    assert 'sesame2' in samples.sample_labels
    assert samples.nb_probes == 9
    assert samples.masks.number_probes_masked(sample_label='sesame1') == 2
    assert samples.masks.number_probes_masked(sample_label='sesame2') == 0
