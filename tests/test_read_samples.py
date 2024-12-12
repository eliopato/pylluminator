import pytest
import os
import numpy as np

from pylluminator.annotations import ArrayType
from pylluminator.samples import read_samples
from pylluminator.utils import download_from_geo

import pandas as pd
pd.options.display.max_columns = 40
pd.options.display.width = 1000

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
    assert my_samples.min_beads == min_beads
    assert len(my_samples.idata ) == max_samples
    assert my_samples.nb_samples == max_samples
    assert len(my_samples._masked_indexes_per_sample) == max_samples
    assert my_samples.annotation.array_type == ArrayType.HUMAN_EPIC_V2

    # Check that the samples are correctly loaded
    sample = my_samples['PREC_500_3']
    assert sample is not None
    assert len(sample) == 937688  # vs 937690 in the original data bc of 2 staining control probes with NA channels

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

    my_samples.calculate_betas()
    my_samples.type1()
    my_samples.type2()
    my_samples.ib()
    my_samples.ib_red()
    my_samples.ib_green()
    my_samples.oob_red()
    my_samples.oob_green()
    my_samples.meth()
    my_samples.unmeth()
    my_samples.type1_red()
    my_samples.type1_green()
    my_samples.cg_probes()
    my_samples.ch_probes()
    my_samples.snp_probes()
    print('ok')
    #
    # assert len(my_samples.type1(False))
    # assert len(my_samples.type2(False))


