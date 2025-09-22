import os
import pandas as pd
import matplotlib.pyplot as plt

from pylluminator.visualizations import (betas_2D, betas_density, dmp_heatmap, nb_probes_per_chr_and_type_hist,
                                         dmr_manhattan_plot, cns_manhattan_plot, visualize_gene, betas_dendrogram,
                                         pc_association_heatmap, pc_correlation_heatmap, methylation_distribution,
                                         betas_heatmap, analyze_replicates, metadata_correlation, metadata_pairplot)

from pylluminator.dm import DM
from pylluminator.cnv import copy_number_segmentation, copy_number_variation


def test_plot_betas_2D(test_samples):
    models = ['PCA', 'MDS', 'DL', 'FA', 'FICA', 'IPCA', 'KPCA', 'LDA', 'MBDL', 'MBNMF', 'MBSPCA', 'NMF', 'SPCA', 'TSVD']
    for m in models:
        betas_2D(test_samples, model=m, nb_probes=1000)

    betas_2D(test_samples, model='PCA', save_path='PCA_2D_plot.png', nb_probes=None, color_column='sample_type', label_column='sample_type')
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

    plt.close("all")

def test_plot_betas_density(test_samples):
    betas_density(test_samples, save_path='betas_plot.png')
    assert os.path.exists('betas_plot.png')
    os.remove('betas_plot.png')

    betas_density(test_samples, save_path='betas_plot.png', title='titre', group_column='sample_type',
                  linestyle_column='sample_type')
    assert os.path.exists('betas_plot.png')
    os.remove('betas_plot.png')

    custom_sheet = test_samples.sample_sheet[test_samples.sample_sheet[test_samples.sample_label_name] == 'LNCAP_500_3']
    betas_density(test_samples, save_path='betas_plot.png', custom_sheet=custom_sheet, apply_mask=False, color_column='sample_type')
    assert os.path.exists('betas_plot.png')
    os.remove('betas_plot.png')

    betas_density(test_samples, save_path='betas_plot.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('betas_plot.png')

    plt.close("all")

def test_betas_heatmap(test_samples):
    betas_heatmap(test_samples, save_path='betas_heatmap.png')
    assert os.path.exists('betas_heatmap.png')
    os.remove('betas_heatmap.png')
    plt.close("all")

def test_dmp_heatmap_ols(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    my_dms = DM(test_samples, '~ sample_type', probe_ids=probe_ids)

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png')
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', contrast=['a', 'b'])
    assert not os.path.exists('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', contrast=my_dms.contrasts[0])
    assert os.path.exists('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', nb_probes=500, figsize=(3, 19), var='sample_type', row_factors=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', drop_na=False, row_factors=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', row_factors=['sample_type'], row_legends=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')

    dmp_heatmap(my_dms, save_path='dmp_heatmap.png', pval_threshold=0.05, effect_size_threshold=0.1)
    assert os.path.exists('dmp_heatmap.png')
    os.remove('dmp_heatmap.png')
    plt.close('all')

def test_dmp_heatmap_mixed_model(test_samples, caplog):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    test_samples.sample_sheet['sentrix_position'] = [name[-1:] for name in test_samples.sample_sheet['sample_name']]
    my_dms = DM(test_samples, '~ sentrix_position', group_column='sentrix_position', probe_ids=probe_ids)

    caplog.clear()
    dmp_heatmap(my_dms, save_path='dmp_heatmap.png')
    assert not os.path.exists('dmp_heatmap.png')
    assert 'You need to specify a contrast for DMPs calculated with a mixed model' in caplog.text

    caplog.clear()
    dmp_heatmap(my_dms, contrast=my_dms.contrasts[0], save_path='dmp_heatmap.png', sort_by='unknown')
    assert not os.path.exists('dmp_heatmap.png')
    assert 'parameter sort_by=unknown not found. Must be "pvalue' in caplog.text

    caplog.clear()
    dmp_heatmap(my_dms, contrast=my_dms.contrasts[0], save_path='dmp_heatmap.png', row_factors=['sample_type'])
    assert os.path.exists('dmp_heatmap.png')
    assert 'ERROR' not in caplog.text
    os.remove('dmp_heatmap.png')

    plt.close('all')


def test_dmr_plot(test_samples):
    probe_ids = test_samples.get_signal_df().reset_index()['probe_id'].sort_values()[:1000].tolist()
    my_dms = DM(test_samples, '~ sample_type', probe_ids=probe_ids)
    my_dms.compute_dmr(probe_ids=probe_ids)

    dmr_manhattan_plot(my_dms, contrast=my_dms.contrasts[0], save_path='dmr_plot.png')
    assert os.path.exists('dmr_plot.png')
    os.remove('dmr_plot.png')

    dmr_manhattan_plot(my_dms,  contrast=my_dms.contrasts[0], save_path='dmr_plot.png', figsize=(3, 19))
    assert os.path.exists('dmr_plot.png')
    os.remove('dmr_plot.png')

    dmr_manhattan_plot(my_dms, contrast=my_dms.contrasts[0], save_path='dmr_plot.png',  title='juju', sig_threshold=0.1)
    assert os.path.exists('dmr_plot.png')
    os.remove('dmr_plot.png')

    plt.close("all")

def test_cns_plot(test_samples):
    cnv_df = copy_number_variation(test_samples, sample_labels='PREC_500_3')
    ranges, signal_bins_df, segments_df = copy_number_segmentation(test_samples, cnv_df, 'PREC_500_3')

    cns_manhattan_plot(signal_bins_df, segments_df, save_path='cns_plot.png')
    assert os.path.exists('cns_plot.png')
    os.remove('cns_plot.png')

    cns_manhattan_plot(signal_bins_df, save_path='cns_plot.png', title='test', figsize=(3, 19))
    assert os.path.exists('cns_plot.png')
    os.remove('cns_plot.png')

    # test wrong parameters
    cns_manhattan_plot(signal_bins_df, segments_df, x_col='tet', save_path='cns_plot.png')
    assert not os.path.exists('cns_plot.png')

    cns_manhattan_plot(signal_bins_df, segments_df, chromosome_col='tet', save_path='cns_plot.png')
    assert not os.path.exists('cns_plot.png')

    cns_manhattan_plot(signal_bins_df, segments_df, y_col='tet', save_path='cns_plot.png')
    assert not os.path.exists('cns_plot.png')

    plt.close("all")

def test_plot_b_chr(test_samples):
    nb_probes_per_chr_and_type_hist(test_samples, save_path='nb_probes_per_chr.png', title='test')
    assert os.path.exists('nb_probes_per_chr.png')
    os.remove('nb_probes_per_chr.png')
    plt.close("all")


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

    plt.close("all")

def test_betas_dendrogram(test_samples):
    betas_dendrogram(test_samples, save_path='dendrogram.png')
    assert os.path.exists('dendrogram.png')
    os.remove('dendrogram.png')

    betas_dendrogram(test_samples, save_path='dendrogram.png', custom_sheet=pd.DataFrame())
    assert not os.path.exists('dendrogram.png')

    betas_dendrogram(test_samples, save_path='dendrogram.png', title='test', apply_mask=False, color_column='sample_type')
    assert os.path.exists('dendrogram.png')
    os.remove('dendrogram.png')

    plt.close("all")

def test_pc_association_heatmap(test_samples):
    pc_association_heatmap(test_samples, ['sample_type', 'sentrix_id', 'sample_number'], save_path='pc_bias.png', nb_probes=1000, n_components=3)
    assert os.path.exists('pc_bias.png')
    os.remove('pc_bias.png')

    pc_association_heatmap(test_samples, save_path='pc_bias.png')
    assert os.path.exists('pc_bias.png')
    os.remove('pc_bias.png')

    # more components than sample, fail
    pc_association_heatmap(test_samples, ['sample_type', 'sentrix_id', 'sample_number'], save_path='pc_bias.png', orientation='h', n_components=8)
    assert not os.path.exists('pc_bias.png')

    plt.close("all")

def test_pc_correlation_heatmap(test_samples, caplog):
    pc_correlation_heatmap(test_samples, ['sample_type', 'sentrix_id', 'sample_number'], save_path='pc_bias.png', nb_probes=1000, n_components=3)
    assert not os.path.exists('pc_bias.png')
    assert 'No significant correlation found' in caplog.text
    assert 'Parameter sentrix_id not found in the sample sheet, skipping' in caplog.text

    pc_correlation_heatmap(test_samples, ['sample_type', 'sample_number'], save_path='pc_bias.png', nb_probes=1000, n_components=3, sig_threshold=None)
    assert os.path.exists('pc_bias.png')
    os.remove('pc_bias.png')

    # more components than sample, fail
    pc_correlation_heatmap(test_samples, ['sample_type', 'sample_number'], save_path='pc_bias.png', orientation='h', n_components=8)
    assert not os.path.exists('pc_bias.png')

    plt.close("all")


def test_methylation_distribution(test_samples, caplog):
    
    # OK
    caplog.clear()
    methylation_distribution(test_samples, group_column='sample_type', save_path='methylation_distribution.png')
    assert os.path.exists('methylation_distribution.png')
    assert 'ERROR' not in caplog.text
    os.remove('methylation_distribution.png')

    # NOK
    caplog.clear()
    methylation_distribution(test_samples, group_column='wrong_col', save_path='methylation_distribution.png')
    assert not os.path.exists('methylation_distribution.png')
    assert 'Column wrong_col not found in the sample sheet' in caplog.text

    # NOK
    caplog.clear()
    methylation_distribution(test_samples, group_column='sample_type', annotation_column='wrong_col', save_path='methylation_distribution.png')
    assert not os.path.exists('methylation_distribution.png')
    assert 'Column wrong_col not found in the annotation data.' in caplog.text

    # NOK
    caplog.clear()
    methylation_distribution(test_samples, group_column='sample_type', delta_beta_threshold=-2, save_path='methylation_distribution.png')
    assert not os.path.exists('methylation_distribution.png')
    assert 'delta_beta_threshold must be betweend 0 and 1' in caplog.text
    caplog.clear()

    # NOK
    methylation_distribution(test_samples, group_column='sample_type', delta_beta_threshold=2, save_path='methylation_distribution.png')
    assert not os.path.exists('methylation_distribution.png')
    assert 'delta_beta_threshold must be betweend 0 and 1' in caplog.text

    plt.close("all")



def test_analyze_replicates(test_samples, caplog):
    test_samples.sample_sheet['replicate'] = ['1', '2', '3', '1', '2', '3']
    caplog.clear()
    analyze_replicates(test_samples, 'replicate', save_path='replicates.png')
    assert 'ERROR' not in caplog.text
    assert os.path.exists('replicates.png')
    os.remove('replicates.png')
    plt.close("all")

def test_metadata_correlation(test_samples, caplog):
    metadata_correlation(test_samples, save_path='metadata_correlation.png')
    assert os.path.exists('metadata_correlation.png')
    os.remove('metadata_correlation.png')

    caplog.clear()
    metadata_correlation(['jui'], save_path='metadata_correlation.png')
    assert not os.path.exists('metadata_correlation.png')
    assert 'input_data must be a Samples object or a pandas DataFrame' in caplog.text

    caplog.clear()
    metadata_correlation(test_samples.sample_sheet, columns=['sample_type', 'sentrix_id', 'sample_number'],
                         abs_corr=False, save_path='metadata_correlation.png')
    assert os.path.exists('metadata_correlation.png')
    assert 'Column sentrix_id not found' in caplog.text
    os.remove('metadata_correlation.png')

    caplog.clear()
    metadata_correlation(test_samples.sample_sheet, columns=['sentrix_id'], abs_corr=False,
                         save_path='metadata_correlation.png')
    assert not os.path.exists('metadata_correlation.png')
    assert 'No valid columns to plot' in caplog.text

    plt.close("all")

def test_metadata_pairplot(test_samples, caplog):
    metadata_pairplot(test_samples, save_path='metadata_pairplot.png')
    assert os.path.exists('metadata_pairplot.png')
    os.remove('metadata_pairplot.png')

    caplog.clear()
    metadata_pairplot(['jui'], save_path='metadata_pairplot.png')
    assert not os.path.exists('metadata_pairplot.png')
    assert 'input_data must be a Samples object or a pandas DataFrame' in caplog.text

    caplog.clear()
    metadata_pairplot(test_samples.sample_sheet, columns=['sample_type', 'sentrix_id', 'sample_number'],
                      save_path='metadata_pairplot.png')
    assert os.path.exists('metadata_pairplot.png')
    assert 'Column sentrix_id not found' in caplog.text
    os.remove('metadata_pairplot.png')

    caplog.clear()
    metadata_pairplot(test_samples.sample_sheet, columns=['sentrix_id'], save_path='metadata_pairplot.png')
    assert not os.path.exists('metadata_pairplot.png')
    assert 'No valid columns to plot' in caplog.text

    caplog.clear()
    metadata_pairplot(test_samples.sample_sheet.drop(columns='sample_number'), save_path='metadata_pairplot.png')
    assert not os.path.exists('metadata_pairplot.png')
    assert 'No numeric columns to plot' in caplog.text

    caplog.clear()
    metadata_pairplot(test_samples, hue='nonexistent', save_path='metadata_pairplot.png')
    assert not os.path.exists('metadata_pairplot.png')
    assert 'Column nonexistent not found' in caplog.text

    plt.close("all")
