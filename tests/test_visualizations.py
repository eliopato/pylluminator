import os
import pandas as pd

from pylluminator.visualizations import (betas_2D, betas_density, plot_dmp_heatmap, plot_nb_probes_and_types_per_chr,
                                         manhattan_plot_dmr, manhattan_plot_cnv, visualize_gene, betas_dendrogram,
                                         plot_pc_correlation, plot_methylation_distribution, plot_betas_heatmap)

from pylluminator.dm import get_dmp, get_dmr
from pylluminator.cnv import copy_number_variation

def test_plot_betas_2D(test_samples):
    models = ['PCA', 'MDS', 'DL', 'FA', 'FICA', 'IPCA', 'KPCA', 'LDA', 'MBDL', 'MBNMF', 'MBSPCA', 'NMF', 'SPCA', 'TSVD']
    for m in models:
        betas_2D(test_samples, model=m, nb_probes=1000)

    betas_2D(test_samples, model='PCA', save_path='PCA_2D_plot.png', nb_probes=1000, color_column='sample_type', label_column='sample_type')
    assert os.path.exists('PCA_2D_plot.png')
    os.remove('PCA_2D_plot.png')

    betas_2D(test_samples, model='PCA', save_path='PCA_2D_plot.png', nb_probes=1000, color_column='egre', label_column='ger')
    assert os.path.exists('PCA_2D_plot.png')
    os.remove('PCA_2D_plot.png')

    betas_2D(test_samples, model='wrongmodel', nb_probes=1000, save_path='PCA_2D_plot.png')
    assert not os.path.exists('PCA_2D_plot.png')

    custom_sheet = test_samples.sample_sheet[test_samples.sample_sheet[test_samples.sample_label_name] == 'LNCAP_500_3']
    betas_2D(test_samples, model='LDA', save_path='PCA_2D_plot.png', title='new title', n_components=5, custom_sheet=custom_sheet)
    assert os.path.exists('PCA_2D_plot.png')
    os.remove('PCA_2D_plot.png')

    betas_2D(test_samples, custom_sheet=pd.DataFrame())
    assert not os.path.exists('PCA_2D_plot.png')

def test_plot_betas_density(test_samples):
    betas_density(test_samples, save_path='betas_plot.png')
    assert os.path.exists('betas_plot.png')
    os.remove('betas_plot.png')

    betas_density(test_samples, n_ind=5, save_path='betas_plot.png', title='titre', group_column='sample_type',
               linestyle_column='sample_type')
    assert os.path.exists('betas_plot.png')
    os.remove('betas_plot.png')

    custom_sheet = test_samples.sample_sheet[test_samples.sample_sheet[test_samples.sample_label_name] == 'LNCAP_500_3']
    betas_density(test_samples, save_path='betas_plot.png', custom_sheet=custom_sheet, apply_mask=False, color_column='sample_type')
    assert os.path.exists('betas_plot.png')
    os.remove('betas_plot.png')

    betas_density(test_samples, save_path='betas_plot.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('betas_plot.png')

def test_plot_betas_heatmap(test_samples):
    plot_betas_heatmap(test_samples, save_path='betas_heatmap.png')
    assert os.path.exists('betas_heatmap.png')
    os.remove('betas_heatmap.png')

def test_dmp_heatmap_ols(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    dmps, contrasts = get_dmp(test_samples, '~ sample_type', probe_ids=probe_ids)

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png')
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', contrast=['a', 'b'])
    assert not os.path.exists('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', contrast=contrasts[0])
    assert os.path.exists('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', nb_probes=500, figsize=(3, 19), var='sample_type', row_factors=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', drop_na=False, row_factors=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', row_factors=['sample_type'], row_legends=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png', pval_threshold=0.05, delta_beta_threshold=0.1)
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

def test_dmp_heatmap_mixed_model(test_samples, caplog):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    test_samples.sample_sheet['sentrix_position'] = [name[-1:] for name in test_samples.sample_sheet['sample_name']]
    dmps, contrasts = get_dmp(test_samples, '~ sentrix_position', group_column='sentrix_position', probe_ids=probe_ids)

    caplog.clear()
    plot_dmp_heatmap(dmps, test_samples, save_path='dmp_heatmap.png')
    assert not os.path.exists('dmp_heatmap.png')
    assert 'You need to specify a contrast for DMPs calculated with a mixed model' in caplog.text

    caplog.clear()
    plot_dmp_heatmap(dmps, test_samples, contrast=contrasts[0], save_path='dmp_heatmap.png', sort_by='unknown')
    assert not os.path.exists('dmp_heatmap.png')
    assert 'parameter unknown not found. Must be pvalue, delta_beta' in caplog.text

    caplog.clear()
    plot_dmp_heatmap(dmps, test_samples, contrast=contrasts[0], save_path='dmp_heatmap.png', row_factors=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    assert 'ERROR' not in caplog.text
    os.remove('dmp_heatmap.png')


def test_dmr_plot(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    dmps, contrasts = get_dmp(test_samples, '~ sample_type', probe_ids=probe_ids)
    dmrs = get_dmr(test_samples, dmps, contrasts, probe_ids=probe_ids)

    manhattan_plot_dmr(dmrs, contrast=contrasts[0], save_path='dmr_plot.png')
    assert os.path.exists('dmr_plot.png')
    os.remove('dmr_plot.png')

    manhattan_plot_dmr(dmrs,  contrast=contrasts[0], save_path='dmr_plot.png', draw_significance=False, figsize=(3, 19))
    assert os.path.exists('dmr_plot.png')
    os.remove('dmr_plot.png')

    manhattan_plot_dmr(dmrs,  contrast=contrasts[0], save_path='dmr_plot.png',  title='juju', medium_threshold=0.1, high_threshold=0.2)
    assert os.path.exists('dmr_plot.png')
    os.remove('dmr_plot.png')

def test_cnv_plot(test_samples):
    ranges, signal_bins_df, segments_df = copy_number_variation(test_samples, sample_label='PREC_500_3')

    manhattan_plot_cnv(signal_bins_df, segments_df, save_path='cnv_plot.png')
    assert os.path.exists('cnv_plot.png')
    os.remove('cnv_plot.png')

    manhattan_plot_cnv(signal_bins_df, save_path='cnv_plot.png', title='test', figsize=(3, 19))
    assert os.path.exists('cnv_plot.png')
    os.remove('cnv_plot.png')

    # test wrong parameters
    manhattan_plot_cnv(signal_bins_df, segments_df, x_col='tet', save_path='cnv_plot.png')
    assert not os.path.exists('cnv_plot.png')

    manhattan_plot_cnv(signal_bins_df, segments_df, chromosome_col='tet', save_path='cnv_plot.png')
    assert not os.path.exists('cnv_plot.png')

    manhattan_plot_cnv(signal_bins_df, segments_df, y_col='tet', save_path='cnv_plot.png')
    assert not os.path.exists('cnv_plot.png')

def test_plot_b_chr(test_samples):
    plot_nb_probes_and_types_per_chr(test_samples, save_path='nb_probes_per_chr.png', title='test')
    assert os.path.exists('nb_probes_per_chr.png')
    os.remove('nb_probes_per_chr.png')

def test_visualize_gene(test_samples):
    visualize_gene(test_samples, 'TUBA1C', save_path='gene_plot.png')
    assert os.path.exists('gene_plot.png')
    os.remove('gene_plot.png')

    visualize_gene(test_samples, 'DUX4', protein_coding_only=False, apply_mask=False, save_path='gene_plot.png')
    assert os.path.exists('gene_plot.png')
    os.remove('gene_plot.png')

    visualize_gene(test_samples, 'DUX4', padding=50, save_path='gene_plot.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('gene_plot.png')

    visualize_gene(test_samples, 'DUX4', save_path='gene_plot.png', keep_na=True, var='sample_type', row_factors=['sample_type'], row_legends=['sample_type'])
    assert os.path.exists('gene_plot.png')
    os.remove('gene_plot.png')

    visualize_gene(test_samples, 'DUX4', save_path='gene_plot.png',  row_factors=['sample_type'], row_legends=['sample_type'])
    assert os.path.exists('gene_plot.png')
    os.remove('gene_plot.png')

def test_betas_dendrogram(test_samples):
    betas_dendrogram(test_samples, save_path='dendrogram.png')
    assert os.path.exists('dendrogram.png')
    os.remove('dendrogram.png')

    betas_dendrogram(test_samples, save_path='dendrogram.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('dendrogram.png')

    betas_dendrogram(test_samples, save_path='dendrogram.png', title='test', apply_mask=False, color_column='sample_type')
    assert os.path.exists('dendrogram.png')
    os.remove('dendrogram.png')

def test_plot_pc_correlation(test_samples):
    plot_pc_correlation(test_samples, ['sample_type', 'sentrix_id', 'sample_number'], save_path='pc_bias.png', nb_probes=1000, n_components=3)
    assert os.path.exists('pc_bias.png')
    os.remove('pc_bias.png')

    # more components than sample, fail
    plot_pc_correlation(test_samples, ['sample_type', 'sentrix_id', 'sample_number'], save_path='pc_bias.png', orientation='h', n_components=8)
    assert not os.path.exists('pc_bias.png')

def test_methylation_distribution(test_samples, caplog):
    plot_methylation_distribution(test_samples, save_path='methylation_distribution.png')
    assert os.path.exists('methylation_distribution.png')
    assert 'ERROR' not in caplog.text
    os.remove('methylation_distribution.png')

    caplog.clear()
    plot_methylation_distribution(test_samples, group_column='wrong_col', save_path='methylation_distribution.png')
    assert not os.path.exists('methylation_distribution.png')
    assert 'Column wrong_col not found in the sample sheet' in caplog.text

    caplog.clear()
    plot_methylation_distribution(test_samples, group_column='sample_type', save_path='methylation_distribution.png')
    assert 'ERROR' not in caplog.text
    assert os.path.exists('methylation_distribution.png')
    os.remove('methylation_distribution.png')