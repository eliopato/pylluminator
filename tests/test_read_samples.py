import pytest
import os
import numpy as np

from pylluminator.annotations import ArrayType
from pylluminator.samples import read_samples
from pylluminator.utils import download_from_geo

@pytest.fixture
def data_path():
    return os.path.expanduser('~/data/pylluminator-utest')

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
        assert os.path.exists(file_path), f"File {file_path} does not exist"

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
    assert my_samples.masks.number_probes_masked(sample_name='PREC_500_3') == 52

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