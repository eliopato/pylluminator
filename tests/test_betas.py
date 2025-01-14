import pytest
import os

from pylluminator.samples import read_samples
from pylluminator.visualizations import plot_betas

@pytest.fixture
def test_samples():
    min_beads = 0
    max_samples = 1
    data_path = os.path.expanduser('~/data/pylluminator-utest')
    return read_samples(data_path, annotation=None, min_beads=min_beads, max_samples=max_samples)

# test beta values on object my_samples
def test_calculate_betas(test_samples):

    betas = test_samples.get_betas(include_out_of_band=False)
    assert betas.xs('cg00002033_TC12', level="probe_id").values == 0.04825291  # Type I green
    assert betas.xs('rs6991394_BC11', level="probe_id").values == 0.50999004  # Type I red
    assert betas.xs('rs9363764_BC21', level="probe_id").values == 0.373386 # Type II

    betas = test_samples.get_betas(include_out_of_band=True)
    assert betas.xs('cg00002033_TC12', level="probe_id").values == 0.07827754  # Type I green
    assert betas.xs('rs6991394_BC11', level="probe_id").values ==  0.51002073  # Type I red
    assert betas.xs('rs9363764_BC21', level="probe_id").values == 0.373386 # Type II


# def test_plot_betas(test_samples):
#     plot_betas(test_samples, color_column='sample_group', mask=True)
#     plot_betas(test_samples, color_column='sample_group', mask=False)
#     plot_betas(test_samples, group_column='sample_group', mask=True)
#     plot_betas(test_samples, group_column=['sample_group', 'sample_name_group'], mask=True)
#     plot_betas(test_samples, color_column='sample_group', group_column=['sample_group', 'sample_name_group'])