"""Methods to plot data, usually from Sample(s) objects or beta values dataframe"""

import os.path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import gridspec

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

from pylluminator.samples import Samples
from pylluminator.annotations import Annotations
from pylluminator.utils import get_chromosome_number, set_level_as_index, get_logger, merge_alt_chromosomes

LOGGER = get_logger()

def _get_colors(sheet: pd.DataFrame, color_column: str | None, group_column: str | list[str] | None = None, cmap_name: str = 'Spectral') -> (list, dict | None):
    """Define the colors to use for each sample, depending on the columns used to categorized them.

    :param sheet: sample sheet data frame
    :type sheet: pandas.DataFrame

    :param color_column: name of the column of the sample sheet to use for color. If None, the function will return empty objects.
    :type color_column: str | None

    :param cmap_name: name of the matplotlib color map to use. Default: spectral
    :type cmap_name: str

    :return: the list of legend handles and a dict of color categories, with keys being the values of color_column
    :rtype: tuple[list,dict | None]"""
    if color_column is None and group_column is None:
        return [], None

    legend_handles = []
    color_categories = dict()
    cmap = colormaps[cmap_name]

    if group_column is not None:
        grouped_sheet = sheet.groupby(group_column)
        nb_colors = len(grouped_sheet)
        for i, (group_name, group) in enumerate(grouped_sheet):
            color_categories[str(group_name).replace("'", "")] = cmap(i / max(1, nb_colors - 1))
    elif color_column == 'sample_name':
        color_categories = {name: cmap(i / len(sheet)) for i, name in enumerate(sheet.sample_name)}
        legend_handles += [Line2D([0], [0], color=color, label=label) for label, color in color_categories.items()]
    else:
        grouped_sheet = sheet.groupby(color_column)
        nb_colors = len(grouped_sheet)
        legend_handles += [Line2D([0], [0], color='black', linestyle='', label=color_column)]
        for i, (group_name, group) in enumerate(grouped_sheet):
            color = cmap(i / max(1, nb_colors - 1))
            for name in group.sample_name:
                color_categories[name] = color
            group_name = str(group_name).replace("'", "").replace('(','').replace(')','')
            legend_handles += [mpatches.Patch(color=color, label=group_name)]
    return legend_handles, color_categories


def _get_linestyles(sheet: pd.DataFrame, column: str | None) -> (list, dict | None):
    """Define the line style to use for each sample, depending on the column used to categorized them.

    :param sheet: sample sheet data frame
    :type sheet: pandas.DataFrame
    :param column: name of the column of the sample sheet to use. If None, the function will return empty objects.
    :type column: str | None

    :return: the list of legend handles and a dict of line styles, with keys being the values of column
    :rtype: tuple[list,dict | None]"""

    if column is None:
        return [], None

    linestyle_categories = dict()
    legend_handles = []
    line_styles = ['-', ':', '--', '-.']

    # determine the line style
    categories = sorted(set(sheet[column]))
    for i, category in enumerate(categories):
        linestyle_categories[category] = line_styles[i % len(line_styles)]
    legend_handles += [Line2D([0], [0], color='black', linestyle='', label=f'{column} :')]
    legend_handles += [Line2D([0], [0], color='black', linestyle=ls, label=label) for label, ls in
                       linestyle_categories.items()]

    return legend_handles, linestyle_categories


def plot_betas(samples: Samples, n_ind: int = 100, title: None | str = None, group_column: None | str | list[str] = None,
               color_column='sample_name',  linestyle_column=None,
               custom_sheet: None | pd.DataFrame = None, mask=True, save_path: None | str=None) -> None:
    """Plot beta values density for each sample

    :param samples: with beta values already calculated
    :type samples: Samples

    :param n_ind: number of evaluation points for the estimated PDF. Default: 100
    :type n_ind: int

    :param title: custom title for the plot to override generated title. Default: None
    :type title: str | None

    :param color_column: name of a Sample Sheet column to define which samples get the same color. Default: sample_name
    :type color_column: str

    :param group_column: compute the average beta values per group of samples. Default: None
    :type group_column: str | list[str] | None

    :param linestyle_column: name of a Sample Sheet column to define which samples get the same line style. Default: None
    :type linestyle_column: str | None

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the samples to display
    :type custom_sheet: pandas.DataFrame

    :param mask: true removes masked probes from betas, False keeps them. Default: True
    :type mask: bool

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str

    :return: None"""

    # initialize values
    plt.style.use('ggplot')
    # get betas with or without masked probes and samples
    betas = samples.get_betas(mask = mask, custom_sheet=custom_sheet)
    sheet = samples.sample_sheet[samples.sample_sheet.sample_name.isin(betas.columns)]

    if group_column is not None:

        grouped_sheet = sheet.groupby(group_column)
        avg_betas_list = []
        group_names = []
        for name, line in grouped_sheet.sample_name.apply(list).items():
            avg_betas_list.append(betas[line].mean(axis=1))
            group_names.append(name)

        betas = pd.concat(avg_betas_list, axis=1)
        betas.columns = group_names

    # define the color and line style of each sample
    c_legend_handles, colors = _get_colors(sheet, color_column, group_column)
    ls_legend_handles, linestyles = _get_linestyles(sheet, linestyle_column)
    legend_handles = c_legend_handles + ls_legend_handles

    if n_ind < 10:
        LOGGER.warning('n_ind is too low, setting it to 10')
        n_ind = 10

    inds = [(n-2)*(1/n_ind) for n in range(1, n_ind+4)]
    betas.plot.density(ind=inds, figsize=(15, 10), color=colors, linestyle=linestyles)


    title = title if title is not None else f'Beta values of {len(betas.columns)} samples on {len(betas):,} probes'
    plt.title(title)

    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

    plt.show()

# # todo
# def plot_betas_per_design(betas: pd.DataFrame, n_bins: int = 100, title: None | str = None, save_path: None | str=None) -> None:
#     """Plot beta values split by Infinium design type (I and II)
#
#     :param betas: dataframe as output from sample[s].get_betas() - rows are probes and columns sample-s
#     :type betas: pandas.DataFrame
#
#     :param n_bins: number of bins to generate the histogram. Default: 100
#     :type n_bins: int
#
#     :param title: custom title for the plot. Default: None
#     :type title: str
#
#     :param save_path: if set, save the graph to save_path. Default: None
#     :type save_path: str | None
#
#     :return: None
#     """
#     for design_type in ['I', 'II']:
#         betas_to_plot = betas.loc[design_type].transpose()
#         for index, row in betas_to_plot.iterrows():
#             histogram_values = np.histogram(row.dropna().values, bins=n_bins, density=False)
#             plt.plot(histogram_values[1][:-1], histogram_values[0], label=index, linewidth=1)
#
#     title = title if title is not None else f'Beta values per design type on {len(betas):,} probes'
#     plt.title(title)
#     plt.legend()
#
#     if save_path is not None:
#         plt.savefig(os.path.expanduser(save_path))
#
#     plt.show()


def betas_mds(samples: Samples, label_column = 'sample_name', color_column: str = 'sample_name',
              nb_probes: int=1000, random_state: int = 42, title: None | str = None, mask=True,
              custom_sheet: None | pd.DataFrame = None, save_path: None | str=None) -> None:
    """Plot samples in 2D space according to their beta distances.

    :param samples : samples to plot
    :type samples: Samples

    :param label_column: name of the column containing the labels
    :type label_column: str | None

    :param color_column: name of a Sample Sheet column used to give samples from the same group the same color. Default: sample_name
    :type color_column: str

    :param nb_probes: number of probes to use for the model. Default: 1000
    :type nb_probes: int

    :param random_state: seed for the MDS model. Assigning a seed makes the graphe reproducible across calls. Default: 42
    :type random_state: int

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param mask: True removes masked probes from betas, False keeps them. Default: True
    :type mask: bool

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    plt.style.use('ggplot')

    # get betas with or without masked probes and samples
    betas = samples.get_betas(mask=mask, custom_sheet=custom_sheet)

    # keep only samples that are both in sample sheet and betas columns
    sheet = samples.sample_sheet[samples.sample_sheet.sample_name.isin(betas.columns)]

    # get betas with the most variance across samples
    betas_variance = np.var(betas, axis=1)
    indexes_most_variance = betas_variance.sort_values(ascending=False)[:nb_probes].index
    betas_most_variance = betas.loc[indexes_most_variance].dropna()  # MDS doesn't support NAs

    # perform MDS
    mds = MDS(n_components=2, random_state=random_state)
    fit = mds.fit_transform(betas_most_variance.T)

    legend_handles, colors_dict = _get_colors(sheet, color_column)

    plt.figure(figsize=(15, 10))
    labels = [sheet.loc[sheet.sample_name == name, label_column].values[0] for name in betas.columns]
    plt.scatter(x=fit[:, 0], y=fit[:, 1], label=labels, c=[colors_dict[label] for label in labels])

    for index, name in enumerate(labels):
        plt.annotate(name, (fit[index, 0], fit[index, 1]), fontsize=9)

    title = title if title is not None else f'MDS of the {nb_probes} most variable probes'
    plt.title(title)

    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

    plt.show()


def betas_dendrogram(samples: Samples, title: None | str = None, color_column: str|None=None, custom_sheet: pd.DataFrame | None = None, mask: bool = True, save_path: None | str=None) -> None:
    """Plot dendrogram of samples according to their beta values distances.

    :param samples: samples to plot
    :type samples: Samples

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param mask: True removes masked probes from betas, False keeps them. Default: True
    :type mask: bool

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 10))

    betas = samples.get_betas(drop_na=True, mask=mask, custom_sheet=custom_sheet)
    sheet = samples.sample_sheet[samples.sample_sheet.sample_name.isin(betas.columns)]

    linkage_matrix = linkage(betas.T.values, optimal_ordering=True, method='complete')
    dendrogram(linkage_matrix, labels=betas.columns, orientation='left')

    if color_column is not None:
        legend_handles, label_colors = _get_colors(sheet, color_column=color_column)

        for lbl in plt.gca().get_ymajorticklabels():
            lbl.set_color(label_colors[lbl.get_text()])

        if len(legend_handles) > 0:
            plt.legend(handles=legend_handles)

    # todo : different handling for > 10 samples ? that's the behaviour in ChAMP:
    # SVD < - svd(beta)
    # rmt.o < - EstDimRMT(beta - rowMeans(beta))
    # k < - rmt.o$dim
    # if (k < 2) k < - 2
    # M < - SVD$v[, 1:k]
    # rownames(M) < - colnames(beta)
    # colnames(M) < - paste("Component", c(1: k))
    # hc < - hclust(dist(M))

    title = title if title is not None else f'Samples\' beta values distances on {len(betas.dropna()):,} probes'
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

    plt.show()


########################################################################################################################


def get_nb_probes_per_chr_and_type(samples: Samples) -> (pd.DataFrame, pd.DataFrame):
    """Count the number of probes covered by the sample-s per chromosome and design type

    :param samples: Samples to analyze
    :type samples: Samples

    :return: None"""

    chromosome_df = pd.DataFrame(columns=['not masked', 'masked'])
    type_df = pd.DataFrame(columns=['not masked', 'masked'])
    manifest = samples.annotation.probe_infos.copy()
    manifest['chromosome'] = merge_alt_chromosomes(manifest['chromosome'])

    # for name, masked in [('not masked', True), ('masked', False)]:
    masked_probes = set()
    for current_sample in samples.sample_names:
        mask = samples.masks.get_mask(sample_name=current_sample)
        if mask is not None:
            masked_probes.update(mask[mask].index.get_level_values('probe_id'))

    unmasked_probes = samples.get_signal_df(False).index.get_level_values('probe_id').difference(masked_probes)

    for name, probes in [('not masked', unmasked_probes), ('masked', masked_probes)]:
        chrm_and_type = manifest.loc[manifest.probe_id.isin(probes), ['probe_id', 'chromosome', 'type']].drop_duplicates()
        chromosome_df[name] = chrm_and_type.groupby('chromosome', observed=True).count()['probe_id']
        type_df[name] = chrm_and_type.groupby('type', observed=False).count()['probe_id']

    # chromosome_df['masked'] = chromosome_df['masked'] - chromosome_df['not masked']
    # type_df['masked'] = type_df['masked'] - type_df['not masked']

    # get the chromosomes numbers to order data frame correctly
    chromosome_df['chr_id'] = get_chromosome_number(chromosome_df.index.tolist())
    chromosome_df = chromosome_df.sort_values('chr_id').drop(columns='chr_id')

    return chromosome_df, type_df


def plot_nb_probes_and_types_per_chr(sample: Samples, title: None | str = None, save_path: None | str=None) -> None:
    """Plot the number of probes covered by the sample per chromosome and design type

    :param sample: Samples to be plotted
    :type sample: Samples

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    chromosome_df, type_df = get_nb_probes_per_chr_and_type(sample)

    fig, axes = plt.subplots(2)

    chromosome_df.plot.bar(stacked=True, figsize=(15, 10), ax=axes[0])
    type_df.plot.bar(stacked=True, figsize=(15, 10), ax=axes[1])

    for container in axes[0].containers:
        axes[0].bar_label(container, label_type='center', rotation=90, fmt='{:,.0f}')

    for container in axes[1].containers:
        axes[1].bar_label(container, label_type='center', fmt='{:,.0f}')

    if title is None:
        title = f'Number of probes per chromosome and type for {sample.nb_samples} samples'

    fig.suptitle(title)

    if save_path is not None:
        fig.savefig(os.path.expanduser(save_path))

########################################################################################################################


def plot_dmp_heatmap(dmps: pd.DataFrame, samples: Samples, contrast: str | None=None,
                     nb_probes: int = 100, figsize=(10,15),
                     var: str | None | list[str] =None, custom_sheet: pd.DataFrame | None=None,
                     drop_na=True, save_path: None | str=None) -> None:
    """Plot a heatmap of the probes that are the most differentially methylated, showing hierarchical clustering of the
    probes with dendrograms on the sides.

    :param dmps:  p-values and statistics for each probe, as returned by get_dmp()
    :type dmps: pandas.DataFrame
    :param samples: samples to use for plotting
    :type samples: Samples
    :param contrast: name of the contrast to use to sort beta values. Must be one of the output contrasts from get_dmp().
        If None is given, will use the F-statistics p-value. Default: None
    :type contrast: str | None
    :param nb_probes: number of probes to plot. Default: 100
    :type nb_probes: int
    :param figsize: size of the plot. Default: (10, 15)
    :type figsize: tuple
    :param var: name of the variable to use for the columns of the heatmap. If None, will use the sample names. Default: None
    :type var: str | list[str] | None
    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None
    :param drop_na: set to True to drop probes with any NA beta values. Note that if set to False, the rendered plot
        won't show the hierarchical clusters. Default: True
    :type drop_na: bool
    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    if dmps is None or len(dmps) == 0:
        return

    if isinstance(contrast, list):
        LOGGER.error('plot_dmp_heatmap() : contrast must be a string, not a list')
        return
    if contrast is None:
        sorted_probes = dmps.sort_values('f_pvalue').index
    else:
        sorted_probes = dmps.sort_values(f'{contrast}_p_value').index

    # sort betas per p-value
    betas = samples.get_betas(custom_sheet=custom_sheet, drop_na=drop_na)
    if var is not None:
        if isinstance(var, str):
            var = [var]
        sheet = samples.sample_sheet
        colnames = [c + ' (' + ', '.join([str(sheet.loc[sheet.sample_name == c, v].iloc[0]) for v in var]) + ')' for c in betas.columns]
        betas.columns = colnames
    betas = set_level_as_index(betas, 'probe_id', drop_others=True)
    sorted_probes = sorted_probes[sorted_probes.isin(betas.index)]
    nb_probes = min(nb_probes, len(sorted_probes))
    sorted_betas = betas.loc[sorted_probes][:nb_probes].T

    if drop_na:
        plot = sns.clustermap(sorted_betas, yticklabels=True, xticklabels=True, figsize=figsize)
        if save_path is not None:
            plot.savefig(os.path.expanduser(save_path))
    else:
        plot = sns.heatmap(sorted_betas.sort_values(betas.columns[0]), yticklabels=True, xticklabels=True, figsize=figsize)
        if save_path is not None:
            plot.get_figure().savefig(os.path.expanduser(save_path))


def _manhattan_plot(data_to_plot: pd.DataFrame, segments_to_plot: pd.DataFrame = None, chromosome_col='chromosome',
                    x_col='start', y_col='p_value', log10=False, figsize=(20,14),
                    annotation: Annotations | None = None, annotation_col: str = 'genes',
                    medium_threshold=1e-05, high_threshold=5e-08,
                    title: None | str = None, draw_significance=False, save_path: None | str=None) -> None:
    """Display a Manhattan plot of the given data.

    :param data_to_plot: dataframe to use for plotting. Typically, a dataframe returned by get_dmrs()
    :type data_to_plot: pandas.DataFrame

    :param segments_to_plot: if set, display the segments using columns "chromosome", "start", "end" and "mean_cnv" of
        the given dataframe, where start and end are the position on the chromosome (as returned by copy_number_variation())
    :type segments_to_plot: pandas.DataFrame

    :param chromosome_col: the name of the chromosome column in the `data_to_plot` dataframe. Default: chromosome
    :type chromosome_col: str

    :param x_col: name of the column to use for X axis, start position of the probe/bin. Default: start
    :type x_col: str

    :param y_col: the name of the value column in the `data_to_plot` dataframe. Default: p_value
    :type y_col: str

    :param annotation: Annotation data to use to annotation significant probes. No annotation if set to None. Default: None
    :type annotation: Annotations | None

    :param annotation_col: the name of a column used to write annotation on the plots for data that is above the
        significant threshold. Must be a column in the Annotation data. Default: None
    :type annotation_col: str | None

    :param medium_threshold: set the threshold used for displaying annotation (and significance line if d
        raw_significance is True). Default: 1e-05
    :type medium_threshold: float

    :param high_threshold: set the threshold for the higher significance line (drawn if draw_significance is True).
        Default: 1e-08
    :type high_threshold: float

    :param log10: apply -log10 on the value column. Default: True
    :type log10: bool

    :param draw_significance: draw p-value significance lines (at 1e-05 and 5e-08). Default: False
    :type draw_significance: bool

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    if data_to_plot is None or len(data_to_plot) == 0:
        return

    # reset index as we might need to use the index as a column (e.g. to annotate probe ids)
    data_to_plot = data_to_plot.reset_index().dropna(subset=y_col)

    data_to_plot['merged_chr'] = merge_alt_chromosomes(data_to_plot[chromosome_col])

    # convert the chromosome column to int values
    if data_to_plot.dtypes[chromosome_col] is not int:
        data_to_plot['chr_id'] = get_chromosome_number(data_to_plot['merged_chr'], True)
        data_to_plot = data_to_plot.astype({'chr_id': 'int'})
    else:
        data_to_plot['chr_id'] = data_to_plot[chromosome_col]

    # sort by chromosome and make the column a category
    data_to_plot = data_to_plot.sort_values(['chr_id', x_col]).astype({'chr_id': 'category'})

    # figure initialization
    fig, ax = plt.subplots(figsize=figsize)
    margin = int(max(data_to_plot[x_col]) / 10)
    chrom_start, chrom_end = 0, 0
    x_labels, x_major_ticks, x_minor_ticks = [], [], [0]

    # apply -log10 to p-values if needed
    if log10:
        data_to_plot[y_col] = -np.log10(data_to_plot[y_col])
        high_threshold = -np.log10(high_threshold)
        medium_threshold = -np.log10(medium_threshold)

    # define colormap and limits
    v_max = np.max(data_to_plot[y_col])
    v_min = np.min(data_to_plot[y_col])
    if v_min < 0:
        cmap = colormaps.get_cmap('jet')
    else:
        cmap = colormaps.get_cmap('viridis').reversed()
        v_min = 0

    # check annotation parameter, and select and clean up annotation if defined
    gene_info = None
    if annotation is not None and annotation_col not in annotation.probe_infos.columns:
        LOGGER.error(f'{annotation_col} was not found in the annotation dataframe. '
                     f'Available columns : {annotation.probe_infos.columns}.')
        annotation = None
    elif annotation is not None:
        gene_info = annotation.probe_infos[['probe_id', annotation_col]].drop_duplicates().set_index('probe_id')
        gene_info.loc[gene_info[annotation_col].isna(), annotation_col] = ''

    # make indices for plotting
    data_to_plot_grouped = data_to_plot.groupby('chr_id', observed=True)

    # plot each chromosome scatter plot with its assigned color
    for num, (name, group) in enumerate(data_to_plot_grouped):
        # add margin to separate a bit the different groups; otherwise small groups won't show
        group[x_col] = chrom_start + group[x_col] + margin
        chrom_end = max(group[x_col]) + margin

        # build the chromosomes scatter plot
        ax.scatter(group[x_col], group[y_col], c=group[y_col], vmin=v_min, vmax=v_max, cmap=cmap, alpha=1)
        # save chromosome's name and limits for x-axis
        x_labels.append(' '.join(set(group['merged_chr'])).replace('chr', ''))
        x_minor_ticks.append(chrom_end)  # chromosome limits
        x_major_ticks.append(chrom_start + (chrom_end - chrom_start) / 2)  # label position]

        # plot segments if a segment df is provided
        if segments_to_plot is not None:
            for chromosomes in set(group['merged_chr']):
                chrom_segments = segments_to_plot[segments_to_plot.chromosome == chromosomes]
                for segment in chrom_segments.itertuples(index=False):
                    plt.plot([chrom_start + segment.start, chrom_start + segment.end],
                             [segment.mean_cnv, segment.mean_cnv],
                             c='black',
                             linewidth=2,
                             alpha=1)

        # draw annotations for probes that are over the threshold, if annotation_col is set
        if annotation is not None:
            if log10:
                indexes_to_annotate = group[y_col] > medium_threshold
            else:
                indexes_to_annotate = group[y_col] < medium_threshold
            x_col_idx = group.columns.get_loc(x_col)
            y_col_idx = group.columns.get_loc(y_col)
            for row in group[indexes_to_annotate].itertuples(index=False):
                gene_name = gene_info.loc[row.probe_id, annotation_col]
                plt.annotate(gene_name, (row[x_col_idx] + 0.03, row[y_col_idx] + 0.03), c=cmap(row[y_col_idx] / v_max))

        chrom_start = chrom_end

    ax.set_facecolor('#EBEBEB')  # set background color to grey
    [ax.spines[side].set_visible(False) for side in ax.spines]  # hide plot frame

    # add lines of significance threshold
    if draw_significance:
        x_start = 0 - margin
        x_end =  chrom_end + margin
        plt.plot([x_start, x_end], [high_threshold, high_threshold], c=cmap(high_threshold), alpha=0.7, ls=':')
        plt.plot([x_start, x_end], [medium_threshold, medium_threshold], ls=':', c=cmap(medium_threshold), alpha=0.5)

    # grids style and plot limits
    ax.xaxis.grid(True, which='minor', color='white', linestyle='--')
    ax.yaxis.grid(True, color='white', alpha=0.9, linestyle='dotted')
    ax.set_axisbelow(True)  # so that the axis lines stay behind the dots
    ax.set_xlim([0 - margin, chrom_end + margin])

    # display chromosomes labels on x axis
    ax.set_xticks(x_major_ticks, labels=x_labels)
    ax.set_xticks(x_minor_ticks, minor=True)  # show ticks for chromosomes limits
    ax.tick_params(axis='x', length=0)  # hide ticks for chromosomes labels
    ax.set_xlabel('chromosome')

    # center vertically on 0 for graphs that include negative values
    if v_min < 0:
        y_lim_inf, y_lim_sup = ax.get_ylim()
        y_lim = max(abs(y_lim_inf), abs(y_lim_sup))
        ax.set_ylim(-y_lim, y_lim)

    # define y label and graph title
    ax.set_ylabel(f'log10({y_col})' if log10 else y_col)

    if title is None:
        what = 'probes' if 'probe_id' in data_to_plot.columns else 'bins'
        title = f'Manhattan plot of {len(data_to_plot):,} {what}'
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

    plt.show()


def manhattan_plot_dmr(data_to_plot: pd.DataFrame, contrast: str,
                       chromosome_col='chromosome', x_col='start', y_col='p_value',
                       annotation: Annotations | None = None, annotation_col='genes', log10=True,
                       draw_significance=True, figsize=(20, 14),
                       medium_threshold=1e-05, high_threshold=5e-08,
                       title: None | str = None, save_path: None | str=None):
    """Display a Manhattan plot of the given DMR data, designed to work with the dataframe returned by get_dmrs()


    :param data_to_plot: dataframe to use for plotting. Typically, a dataframe returned by get_dmrs()
    :type data_to_plot: pandas.DataFrame

    :param chromosome_col: the name of the chromosome column in the `data_to_plot` dataframe. Default: chromosome
    :type chromosome_col: str

    :param x_col: name of the column to use for X axis, start position of the probe/bin. Default: start
    :type x_col: str

    :param y_col: the name of the value column in the `data_to_plot` dataframe. Default: p_value
    :type y_col: str

    :param annotation: Annotation data to use to annotation significant probes. No annotation if set to None. Default: None
    :type annotation: Annotations | None

    :param annotation_col: the name of a column used to write annotation on the plots for data that is above the
        significant threshold. Must be a column in the Annotation data. Default: None
    :type annotation_col: str | None

    :param medium_threshold: set the threshold used for displaying annotation (and significance line if
        raw_significance is True). Default: 1e-05
    :type medium_threshold: float

    :param high_threshold: set the threshold for the higher significance line (drawn if draw_significance is True).
        Default: 1e-08
    :type high_threshold: float

    :param log10: apply -log10 on the value column. Default: True
    :type log10: bool

    :param draw_significance: draw p-value significance lines (at 1e-05 and 5e-08). Default: False
    :type draw_significance: bool

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None
    :return: nothing"""

    _manhattan_plot(data_to_plot=data_to_plot, chromosome_col=chromosome_col, y_col=f'{contrast}_{y_col}', x_col=x_col,
                    draw_significance=draw_significance, annotation=annotation, annotation_col=annotation_col,
                    medium_threshold=medium_threshold, high_threshold=high_threshold, figsize=figsize,
                    log10=log10, title=title, save_path=save_path)


def manhattan_plot_cnv(data_to_plot: pd.DataFrame, segments_to_plot=None,
                       x_col='start_bin', chromosome_col='chromosome', y_col='cnv',
                       figsize=(20, 14),
                       title: None | str = None, save_path: None | str=None) -> None:
    """Display a Manhattan plot of the given CNV data, designed to work with the dataframes returned by
    copy_number_variation()

    :param data_to_plot: dataframe to use for plotting. Typically, a dataframe returned by get_dmrs()
    :type data_to_plot: pandas.DataFrame

    :param segments_to_plot: if set, display the segments using columns "chromosome", "start", "end" and "mean_cnv" of
        the given dataframe, where start and end are the position on the chromosome (as returned by copy_number_variation())
    :type segments_to_plot: pandas.DataFrame

    :param chromosome_col: the name of the chromosome column in the `data_to_plot` dataframe. Default: chromosome
    :type chromosome_col: str

    :param x_col: name of the column to use for X axis, start position of the probe/bin. Default: start_bin
    :type x_col: str

    :param y_col: the name of the value column in the `data_to_plot` dataframe. Default: cnv
    :type y_col: str

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    _manhattan_plot(data_to_plot=data_to_plot, segments_to_plot=segments_to_plot, x_col=x_col,
                    chromosome_col=chromosome_col, y_col=y_col, title=title, figsize=figsize,
                    log10=False, annotation=None, draw_significance=False, save_path=save_path)

########################################################################################################################

def visualize_gene(samples: Samples, gene_name: str, mask: bool=True, padding=1500, keep_na: bool=False,
                   protein_coding_only=True, custom_sheet: pd.DataFrame | None=None, var: None | str | list[str] = None,
                   figsize=(20, 20), save_path: None | str=None) -> None:
    """Show the beta values of a gene for all probes and samples in its transcription zone.

    :param samples : samples with beta values already calculated
    :type samples: Samples

    :param gene_name : name of the gene to visualize
    :type gene_name: str

    :param mask: True removes masked probes from betas, False keeps them. Default: True
    :type mask: bool

    :param padding: length in kb pairs to add at the end and beginning of the transcription zone. Default: 1500
    :type: int

    :param keep_na : set to True to only output probes with no NA value for any sample. Default: False
    :type keep_na: bool

    :param protein_coding_only: limit displayed transcripts to protein coding ones. Default: True
    :type protein_coding_only: bool

    :param figsize: size of the whole plot. Default: (20, 20)
    :type figsize: tuple[int, int]

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    color_map = {'gneg': 'lightgrey', 'gpos25': 'lightblue', 'gpos50': 'blue', 'gpos75': 'darkblue', 'gpos100': 'purple',
                 'gvar': 'lightgreen', 'acen': 'yellow', 'stalk': 'pink'}

    links_args = {'color': 'red', 'alpha': 0.3}

    ################## DATA PREP
    genome_info = samples.annotation.genome_info

    transcript_data = genome_info.transcripts_exons
    gene_data = transcript_data[transcript_data.gene_name == gene_name]
    if protein_coding_only:
        gene_data = gene_data[gene_data.transcript_type == 'protein_coding']

    if len(gene_data) == 0:
        LOGGER.error(f'Gene {gene_name} not found in the annotation data.')
        return

    chromosome = str(gene_data.iloc[0].chromosome)
    chr_df = genome_info.chromosome_regions.loc[chromosome]
    gene_transcript_start = gene_data.transcript_start.min() - padding
    gene_transcript_end = gene_data.transcript_end.max() + padding
    gene_transcript_length = gene_transcript_end - gene_transcript_start

    txns = genome_info.transcripts_exons
    gene_data = txns[(txns.chromosome == chromosome)
                     &
                     (((txns.transcript_start >= gene_transcript_start) & (txns.transcript_start <= gene_transcript_end))
                      | ((txns.transcript_end >= gene_transcript_start) & (txns.transcript_end <= gene_transcript_end)))
                     & (txns.transcript_type == 'protein_coding')]

    gene_data = gene_data.sort_values('transcript_start')

    probe_info_df = samples.annotation.probe_infos
    is_gene_in_interval = probe_info_df.chromosome == chromosome.replace('chr', '')
    is_gene_in_interval &= (probe_info_df.start >= gene_transcript_start) & (probe_info_df.start <= gene_transcript_end)
    is_gene_in_interval &= (probe_info_df.end >= gene_transcript_start) & (probe_info_df.end <= gene_transcript_end)
    gene_probes = probe_info_df[is_gene_in_interval][['probe_id', 'start', 'end']].drop_duplicates().set_index('probe_id')
    gene_betas = samples.get_betas(mask=mask, custom_sheet=custom_sheet)
    gene_betas = set_level_as_index(gene_betas, 'probe_id', drop_others=True)
    betas_location = gene_betas.join(gene_probes, how='inner').sort_values('start')

    print(f'chromosome {chromosome}, pos {gene_transcript_start} - {gene_transcript_end}')

    ################## PLOT LINKS BETWEEN TRANSCRIPTS AND BETAS

    # chromosome, chr-transcript links, transcripts, transcript-betas lings, betas heatmap
    height_ratios = [0.05, 0.05, 0.45, 0.05, 0.4]
    nb_plots = len(height_ratios)

    betas_data = betas_location if keep_na else betas_location.dropna()

    heatmap_data = betas_data.drop(columns=['start', 'end']).T

    if len(heatmap_data) == 0:
        LOGGER.error('no beta data to plot')
        return

    # add variable values to the column names
    if var is not None:
        if isinstance(var, str):
            var = [var]
        sheet = samples.sample_sheet
        colnames = [c + ' (' + ', '.join([str(sheet.loc[sheet.sample_name == c, v].iloc[0]) for v in var]) + ')' for c in heatmap_data.index]
        heatmap_data.index = colnames

    if keep_na:
        fig, axes = plt.subplots(figsize=figsize, nrows=nb_plots, height_ratios=height_ratios)
        sns.heatmap(heatmap_data, ax=axes[-1], cbar=False, xticklabels=True, yticklabels=True)
    else:
        dendrogram_ratio = 0.05
        g = sns.clustermap(heatmap_data, figsize=figsize, cbar_pos=None, col_cluster=False,
                           dendrogram_ratio=dendrogram_ratio, xticklabels=True, yticklabels=True)
        shift_ratio = np.sum(height_ratios[:-1])
        g.gs.update(top=shift_ratio)  # shift the heatmap to the bottom of the figure
        gs2 = gridspec.GridSpec(nb_plots - 1, 1, left=dendrogram_ratio + 0.005, bottom=shift_ratio, height_ratios=height_ratios[:-1])
        axes = [g.fig.add_subplot(gs) for gs in gs2]

    ################## plot chromosome and chromosome-transcript links

    chr_ax = axes[0]

    # make rectangles of different colors depending on the chromosome region
    for _, row in chr_df.iterrows():
        chr_ax.add_patch(mpatches.Rectangle((row.start, 0), row.end - row.start, 0.5, color=color_map[row.giemsa_staining]))

    # red lines showing the beginning and end of the gene region
    chr_ax.plot([gene_transcript_start, gene_transcript_start], [-1, 1], **links_args)
    chr_ax.plot([gene_transcript_end, gene_transcript_end], [-1, 1], **links_args)

    chr_length = chr_df['end'].max()
    chr_ax.set_xlim(0, chr_length)
    chr_ax.set_ylim(0, 0.5)
    chr_ax.axis('off')

    ################## PLOT TRANSCRIPTS

    trans_ax = axes[2]
    y_labels = []
    y_positions = []
    transcript_index = 0

    for transcript_index, (transcript_id, transcript_data) in enumerate(gene_data.groupby('transcript_id', sort=False)):

        # name of the transcript for y ticks labels
        y_labels.append(transcript_id)
        y_position = transcript_index * 2 + 0.5
        y_positions.append(y_position)

        transcript_start = transcript_data.transcript_start.min()
        transcript_end = transcript_data.transcript_end.max()

        if transcript_data.iloc[0].transcript_strand == '-':
            arrow_coords = (transcript_start, y_position, -padding, 0)
            transcript_end += padding
        else:
            arrow_coords = (transcript_end, y_position, padding, 0)
            transcript_start -= padding

        # line of the full transcript length
        trans_ax.plot([transcript_start, transcript_end], [y_position, y_position], color='black', alpha=0.3, zorder=1)
        # arrow at the end of the line to show the strand direction
        trans_ax.arrow(*arrow_coords, shape='full', fill=True, color='black', head_width=0.75, length_includes_head=True,
                    head_length=int(padding / 3), width=0, alpha=0.3)

        # draw the patches for each transcript location
        for row in transcript_data.itertuples():
            trans_ax.add_patch(mpatches.Rectangle((row.transcript_start, y_position-0.5),
                                                  row.transcript_end - row.transcript_start, 1, color='black'))

            # # if a probe intersects with a transcript, draw a colored patch
            # for beta_row in betas_data.itertuples():
            #     if (row.start <= beta_row.start <= row.End) or (row.Start <= beta_row.end <= row.End):
            #         rec_coords = (beta_row.start, i*3+0.2), beta_row.end - beta_row.start
            #         trans_ax.add_patch(mpatches.Rectangle(*rec_coords, 0.6, color='limegreen', zorder=2))


    chr_trans_link_ax = axes[1]

    for beta_row in betas_data.itertuples():
        rec_coords = (beta_row.start, 0), beta_row.end - beta_row.start, transcript_index * 2 + 1
        trans_ax.add_patch(mpatches.Rectangle(*rec_coords, zorder=2, **links_args))
        # draw link between chromosome and transcript
        trans_position = beta_row.start + (beta_row.end - beta_row.start) / 2
        trans_position_in_chr = trans_position / chr_length
        trans_position_in_trans = (trans_position - gene_transcript_start) / gene_transcript_length
        chr_trans_link_ax.plot([trans_position_in_trans, trans_position_in_chr, trans_position_in_chr], [0, 0.8, 1], **links_args)

    # hide all axes but keep the y labels (transcript names)
    [trans_ax.spines[pos].set_visible(False) for pos in ['top', 'right', 'bottom', 'left']]
    trans_ax.set_yticks(y_positions, y_labels)
    trans_ax.set_xticks([])
    trans_ax.set_xlim(gene_transcript_start, gene_transcript_end)
    trans_ax.set_ylim(0, transcript_index * 2 + 1)

    chr_trans_link_ax.set_xlim(0, 1)
    chr_trans_link_ax.set_ylim(0, 1)
    chr_trans_link_ax.axis('off')

    ################## PLOT LINKS

    lin_ax = axes[3]

    nb_probes = len(betas_data)
    probe_shift = 1 / (2 * nb_probes)

    for i, beta_row in enumerate(betas_data.itertuples()):
        probe_loc = beta_row.start - gene_transcript_start + (beta_row.end - beta_row.start) / 2
        x_transcript = probe_loc / gene_transcript_length
        x_beta = i / nb_probes + probe_shift
        lin_ax.plot([x_beta, x_transcript, x_transcript], [0, 1.5, 2], **links_args)

    lin_ax.set_xlim(0, 1)
    lin_ax.set_ylim(0, 2)
    lin_ax.axis('off')

    # for ax in axes[:nb_plots-1]:
    #     ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

    plt.show()