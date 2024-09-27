import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

from illuminator.sample import Sample
from illuminator.samples import Samples
from illuminator.annotations import Annotations
from illuminator.utils import get_chromosome_number, set_level_as_index, get_logger


LOGGER = get_logger()

def _get_colors(sheet: pd.DataFrame, color_column: str | None, color_group_column: str | None, cmap_name: str = 'Spectral'):
    """Define the colors to use for each sample, depending on the columns used to categorized them."""
    legend_handles = []
    color_categories = dict()
    cmap = colormaps[cmap_name]

    # determine the line color
    if color_column is not None:
        categories = sorted(set(sheet[color_column]))
        nb_colors = len(categories)
        if color_group_column is None:
            for i, category in enumerate(categories):
                color_categories[category] = cmap(i / (nb_colors - 1))
        else:
            grouped_sheet = sheet.groupby(color_group_column)
            nb_categories = len(grouped_sheet)
            # gap between colors of two different group
            cmap_group_interval_size = min(0.2, 1 / (nb_categories * 2 - 1))
            # gap between two colors of the same group
            cmap_color_interval_size = (1 - (nb_categories - 1) * cmap_group_interval_size) / nb_colors
            idx_color = 0
            for _, sub_sheet in grouped_sheet:
                categories = sorted(set(sub_sheet[color_column]))
                for i, category in enumerate(categories):
                    idx_color += cmap_color_interval_size
                    color_categories[category] = cmap(idx_color)
                idx_color += cmap_group_interval_size

        # make the legend (title + colors)
        legend_handles += [Line2D([0], [0], color='black', linestyle='', label=f'{color_column} :')]
        legend_handles += [mpatches.Patch(color=color, label=label) for label, color in color_categories.items()]

    return legend_handles, color_categories

def _get_linestyles(sheet: pd.DataFrame, column: str | None):
    """Define the line style to use for each sample, depending on the column used to categorized them."""

    linestyle_categories = dict()
    legend_handles = []
    line_styles = ['-', ':', '--', '-.']

    # determine the line style
    if column is not None:
        categories = sorted(set(sheet[column]))
        for i, category in enumerate(categories):
            linestyle_categories[category] = line_styles[i % len(line_styles)]
        legend_handles += [Line2D([0], [0], color='black', linestyle='', label=f'{column} :')]
        legend_handles += [Line2D([0], [0], color='black', linestyle=ls, label=label) for label, ls in
                           linestyle_categories.items()]

    return legend_handles, linestyle_categories


def plot_betas(samples: Samples, n_bins: int = 100, title: None | str = None,
               color_column='sample_name', color_group_column: None | str = None, linestyle_column=None,
               custom_sheet: None | pd.DataFrame = None, mask=True) -> None:
    """Plot betas values density for each sample

    :param samples : (Samples) with betas already calculated
    :param n_bins: (int, optional, default 100) number of bins to generate the histogram
    :param color_column: (str, optional, default None) name of a Sample Sheet column to define which samples get the
        same color
    :param color_group_column: (str, optional, default None) name of a Sample Sheet column to categorize samples and
        give samples from the same category a similar color shade.
    :param linestyle_column: (str, optional, default None) name of a Sample Sheet column to define which samples get the
        same line style
    :param custom_sheet: (pd.DataFrame, optional) a sample sheet to use. By default, use the samples' sheet. Useful if
        you want to filter the samples to display
    :param title: (str, optional) custom title for plot
    :param mask: (bool, optional, default True) True removes masked probes from betas, False keeps them."

    :return: None"""

    # initialize values
    plt.style.use('ggplot')
    sheet = samples.sample_sheet if custom_sheet is None else custom_sheet
    betas = samples.betas(mask)  # get betas with or without masked probes

    # keep only samples that are both in sample sheet and betas columns
    filtered_samples = [col for col in sheet.sample_name.values if col in betas.columns]
    betas = betas[filtered_samples]
    sheet = sheet[sheet.sample_name.isin(filtered_samples)]

    # define the color and line style of each sample
    c_legend_handles, colors = _get_colors(sheet, color_column, color_group_column)
    ls_legend_handles, linestyles = _get_linestyles(sheet, linestyle_column)
    legend_handles = c_legend_handles + ls_legend_handles

    # plot the data
    plt.figure(figsize=(15, 10))

    color = None
    linestyle = None
    for label, row in betas.transpose().iterrows():
        histogram_y, histogram_x = np.histogram(row.dropna().values, bins=n_bins, density=True)
        sample_sheet_row = sheet[sheet.sample_name == label]
        if color_column is not None:
            label = sample_sheet_row[color_column].iloc[0]
            color = colors[label]
        if linestyle_column is not None:
            linestyle = linestyles[sample_sheet_row[linestyle_column].iloc[0]]
        plt.plot(histogram_x[:-1], histogram_y, label=label, linewidth=1, color=color, linestyle=linestyle)

    title = title if title is not None else f'Beta values of {len(betas.columns)} samples on {len(betas)} probes'
    plt.title(title)

    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()


def plot_betas_grouped(samples: Samples, group_columns: list[str], n_bins: int=100, title: str | None=None, mask=True,
                       custom_sheet: None | pd.DataFrame = None) -> None:
    """Plot betas grouped by one or several sample sheet columns. Display the average beta values per group with a plain
    line, and individual beta values distribution as transparent lines.

    :param samples : (Samples) with betas already calculated
    :param group_columns: (list of str) name of one or several Sample Sheet column to categorize samples and
        give samples from the same category a similar color shade.
    :param n_bins: (int, optional, default 100) number of bins to generate the histogram
    :param title: (str, optional) custom title for plot
    :param mask: (bool, optional, default True) True removes masked probes from betas, False keeps them.
    :param custom_sheet: (pd.DataFrame, optional) a sample sheet to use. By default, use the samples' sheet. Useful if
        you want to filter the samples to display

    :return None"""

    cmap = colormaps['Spectral']
    plt.style.use('ggplot')
    bins = [n*1/n_bins for n in range(0, n_bins+1)]

    sheet = samples.sample_sheet if custom_sheet is None else custom_sheet
    betas = samples.betas(mask)  # get betas with or without masked probes

    # keep only samples that are both in sample sheet and betas columns
    filtered_samples = [col for col in sheet.sample_name.values if col in betas.columns]
    betas = betas[filtered_samples]
    sheet = sheet[sheet.sample_name.isin(filtered_samples)]

    grouped_sheet = sheet.groupby(group_columns)
    nb_groups = len(grouped_sheet)

    plt.figure(figsize=(15, 10))

    for i_group, (group_name, sub_sheet) in enumerate(grouped_sheet):
        color = cmap(i_group / (nb_groups - 1))

        histos = np.zeros((len(sub_sheet), n_bins))

        # draw each sample's beta distribution with a very transparent line, and save values to calculate the average
        for i, sample in enumerate(sub_sheet.sample_name):
            hist_y, _ = np.histogram(betas[sample].dropna().values, bins=bins, density=True)
            plt.plot(bins[:-1], hist_y, linewidth=1, alpha=0.1, color=color)
            histos[i] = hist_y

        # get average histogram for the group
        mean_histo =  histos.mean(axis=0)

        plt.plot(bins[:-1], mean_histo, label=group_name, linewidth=1, color=color)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title=group_columns)

    if title is None:
        title = f'Beta values of {len(betas.columns)} samples on {len(betas)} probes, grouped by {group_columns}'
    plt.title(title)

    plt.show()


def plot_betas_per_design(betas: pd.DataFrame, n_bins: int = 100, title: None | str = None) -> None:

    for design_type in ['I', 'II']:
        betas_to_plot = betas.loc[design_type].transpose()
        for index, row in betas_to_plot.iterrows():
            histogram_values = np.histogram(row.dropna().values, bins=n_bins, density=False)
            plt.plot(histogram_values[1][:-1], histogram_values[0], label=index, linewidth=1)

    title = title if title is not None else f'Samples\' beta values distances on {len(betas)} probes'
    plt.title(title)
    plt.legend()


def betas_mds(samples: Samples, color_group_column: str | None = None, random_state: int = 42, title: None | str = None,
              mask=True, custom_sheet: None | pd.DataFrame = None) -> None:
    """Plot samples in 2D space according to their beta distances.

    :param samples : Samples object, with betas already calculated
    :param color_group_column: (str, optional, default None) name of a Sample Sheet column to categorize samples and
        give samples from the same category a similar color shade.
    :param random_state: (int, optional, default 42) seed for the MDS model. Assigning a seed makes the graphe
        reproducible across calls.
    :param title: (str, optional) custom title for plot
    :param mask: (bool, optional, default True) True removes masked probes from betas, False keeps them.
    :param custom_sheet: (pd.DataFrame, optional) a sample sheet to use. By default, use the samples' sheet. Useful if
        you want to filter the samples to display

    :return: None"""

    sheet = samples.sample_sheet if custom_sheet is None else custom_sheet
    betas = samples.betas(mask)  # get betas with or without masked probes

    # keep only samples that are both in sample sheet and betas columns
    filtered_samples = [col for col in sheet.sample_name.values if col in betas.columns]
    betas = betas[filtered_samples]
    sheet = sheet[sheet.sample_name.isin(filtered_samples)]

    # get betas with the most variance across samples (top 1000)
    betas_variance = np.var(betas, axis=1)
    indexes_most_variance = betas_variance.sort_values(ascending=False)[:1000].index
    betas_most_variance = betas.loc[indexes_most_variance].dropna()  # MDS doesn't support NAs

    # perform MDS
    mds = MDS(n_components=2, random_state=random_state)
    fit = mds.fit_transform(betas_most_variance.T)

    legend_handles, colors_dict = _get_colors(sheet, 'sample_name', color_group_column)
    colors = [colors_dict[sample] for sample in betas.columns]

    plt.figure(figsize=(15, 10))
    plt.scatter(x=fit[:, 0], y=fit[:, 1], label=betas.columns, c=colors)

    for index, name in enumerate(betas.columns):
        plt.annotate(name, (fit[index, 0], fit[index, 1]), fontsize=9)

    title = title if title is not None else f'MDS of the 1000 most variable probes'
    plt.title(title)

    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))


def betas_dendrogram(betas: pd.DataFrame, title: None | str = None) -> None:
    """Plot dendrogram of samples according to their beta values distances.

    :param betas: pd.DataFrame as output from sample[s].get_betas() - rows are probes and columns sample-s
    :param title: custom title for plot

    :return: None"""

    plt.figure(figsize=(15, 10))

    linkage_matrix = linkage(betas.dropna().T.values, optimal_ordering=True, method='complete')
    dendrogram(linkage_matrix, labels=betas.columns, orientation='left')

    # todo : different handling for > 10 samples ? that's the behaviour in ChAMP:
    # SVD < - svd(beta)
    # rmt.o < - EstDimRMT(beta - rowMeans(beta))
    # k < - rmt.o$dim
    # if (k < 2) k < - 2
    # M < - SVD$v[, 1:k]
    # rownames(M) < - colnames(beta)
    # colnames(M) < - paste("Component", c(1: k))
    # hc < - hclust(dist(M))

    title = title if title is not None else f'Samples\' beta values distances on {len(betas.dropna())} probes'
    plt.title(title)

########################################################################################################################


def get_nb_probes_per_chr_and_type(sample: Sample | Samples) -> (pd.DataFrame, pd.DataFrame):
    """Count the number of probes covered by the sample-s per chromosome and design type
    :param sample: Sample or Samples to analyze
    :return: None"""

    chromosome_df = pd.DataFrame(columns=['not masked', 'masked'])
    type_df = pd.DataFrame(columns=['not masked', 'masked'])
    manifest = sample.annotation.probe_infos
    probes = set()

    for name, masked in [('not masked', True), ('masked', False)]:
        if isinstance(sample, Sample):
            probes = sample.get_signal_df(masked).reset_index().probe_id
        elif isinstance(sample, Samples):
            for current_sample in sample.samples.values():
                probes.update(current_sample.get_signal_df(masked).reset_index().probe_id)
        chrm_and_type = manifest.loc[manifest.probe_id.isin(probes), ['probe_id', 'cpg_chrm', 'type']].drop_duplicates()
        chromosome_df[name] = chrm_and_type.groupby('cpg_chrm').count()['probe_id']
        type_df[name] = chrm_and_type.groupby('type', observed=False).count()['probe_id']

    chromosome_df['masked'] = chromosome_df['masked'] - chromosome_df['not masked']
    type_df['masked'] = type_df['masked'] - type_df['not masked']

    # get the chromosomes numbers to order data frame correctly
    chromosome_df['chr_id'] = get_chromosome_number(chromosome_df.index.tolist())
    chromosome_df = chromosome_df.sort_values('chr_id').drop(columns='chr_id')

    return chromosome_df, type_df


def plot_nb_probes_and_types_per_chr(sample: Sample | Samples, title: None | str = None) -> None:
    """Plot the number of probes covered by the sample per chromosome and design type

    :param sample: Sample or Samples to be plotted
    :param title: custom title for plot

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
        if isinstance(sample, Sample):
            title = f'Number of probes per chromosome and type for sample {sample}'
        if isinstance(sample, Samples):
            title = f'Number of probes per chromosome and type for {sample.nb_samples} samples'

    fig.suptitle(title)

########################################################################################################################


def plot_dmp_heatmap(dmp: pd.DataFrame, betas: pd.DataFrame, nb_probes: int = 100, keep_na=False) -> None:
    """Plot a heatmap of the probes that are the most differentially methylated, showing hierarchical clustering of the
    probes with dendrograms on the sides.

    :param dmp: (pd.DataFrame) p-values and statistics for each probe, as returned by get_dmp()
    :param betas: (pd.DataFrame) beta values as output from sample[s].get_betas() rows are probes and columns sample-s
    :param nb_probes: (optional, int) number of probes to plot (default is 100)
    :param keep_na: (optional, bool) set to False to drop probes with any NA beta values (default to False). Note that
        if set to True, the rendered plot won't show the hierarchical clusters

    :return: None"""

    # sort betas per p-value
    sorted_probes = dmp.sort_values('p_value').index
    sorted_betas = set_level_as_index(betas, 'probe_id', drop_others=True).loc[sorted_probes]

    if keep_na:
        sns.heatmap(sorted_betas[:nb_probes].sort_values(betas.columns[0]), xticklabels=True)
    else:
        sns.clustermap(sorted_betas.dropna()[:nb_probes], xticklabels=True)


def _manhattan_plot(data_to_plot: pd.DataFrame, segments_to_plot: pd.DataFrame = None, chromosome_col='Chromosome',
                    x_col='Start', y_col='p_value', log10=False,
                    annotation: Annotations | None = None, annotation_col: str = 'genes',
                    medium_threshold=1e-05, high_threshold=5e-08,
                    title: None | str = None, draw_significance=False) -> None:
    """Display a Manhattan plot of the given data.

    :param data_to_plot: (pd.DataFrame) dataframe to use for plotting. Typically, a dataframe returned by get_dmrs()
    :param segments_to_plot: (optional, pd.DataFrame) if set, display the segments using columns "chromosome", "start",
        "end" and "mean_cnv" of the given dataframe, where start and end are the position on the chromosome (as returned
        by copy_number_variation())
    :param chromosome_col: (optional, string, default 'Chromosome') the name of the Chromosome column in the
        `data_to_plot` dataframe.
    :param x_col: (option, string, default 'Start') name of the column to use for X axis, start position of the probe/bin
    :param y_col: (optional, string, default 'p_value') the name of the value column in the `data_to_plot` dataframe
    :param annotation: (optional, Annotation, default None) Annotation data to use to annotation significant probes.
        Can be None to remove any annotation.
    :param annotation_col: (optional, str, default None) the name of a column used to write annotation on the
        plots for data that is above the significant threshold. Must be a column in the Annotation data
    :param medium_threshold: (optional, float, default 1e-05) set the threshold used for displaying annotation
        (and significance line if draw_significance is True)
    :param high_threshold: (optional, float, default 1e-08) set the threshold for the higher significance line (drawn if
        draw_significance is True)
    :param log10: (optional, boolean, default True) apply -log10 on the value column
    :param draw_significance: (option, boolean, default False) draw p-value significance lines (at 1e-05 and 5e-08)
    :param title: (optional, string) custom title for plot

    :return: nothing"""

    # reset index as we might need to use the index as a column (e.g. to annotate probe ids)
    data_to_plot = data_to_plot.reset_index().dropna(subset=y_col)

    # convert the chromosome column to int values
    if data_to_plot.dtypes[chromosome_col] != int:
        data_to_plot['chr_id'] = get_chromosome_number(data_to_plot[chromosome_col], True)
        data_to_plot = data_to_plot.astype({'chr_id': 'int'})
    else:
        data_to_plot['chr_id'] = data_to_plot[chromosome_col]

    # sort by chromosome and make the column a category
    data_to_plot = data_to_plot.sort_values(['chr_id', x_col]).astype({'chr_id': 'category'})

    # figure initialization
    fig, ax = plt.subplots(figsize=(20, 14))
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
        cmap = colormaps.get_cmap('gist_rainbow')
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
        ax.scatter(group[x_col], group[y_col], c=group[y_col], vmin=v_min, vmax=v_max, cmap=cmap, alpha=0.9)
        # save chromosome's name and limits for x-axis
        x_labels.append(' '.join(set(group[chromosome_col])).replace('chr', ''))
        x_minor_ticks.append(chrom_end)  # chromosome limits
        x_major_ticks.append(chrom_start + (chrom_end - chrom_start) / 2)  # label position]

        # plot segments if a segment df is provided
        if segments_to_plot is not None:
            for chromosomes in set(group[chromosome_col]):
                chrom_segments = segments_to_plot[segments_to_plot.chromosome == chromosomes]
                for segment in chrom_segments.itertuples(index=False):
                    plt.plot([chrom_start + segment.start, chrom_start + segment.end],
                             [segment.mean_cnv, segment.mean_cnv],
                             c=cmap(-segment.mean_cnv),
                             linewidth=2)

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
    ax.set_xlabel('Chromosome')

    # define y label and graph title
    ax.set_ylabel(f'log10({y_col})' if log10 else y_col)
    if title is None:
        if 'probe_id' in data_to_plot.columns:
            title = f'Manhattan plot of {len(data_to_plot)} probes'
        else:
            title = f'Manhattan plot of {len(data_to_plot)} bins'
    plt.title(title)
    plt.show()


def manhattan_plot_dmr(data_to_plot: pd.DataFrame, chromosome_col='Chromosome', x_col='Start', y_col='p_value',
                       annotation: Annotations | None = None, annotation_col='genes', log10=True,
                       draw_significance=True,
                       medium_threshold=1e-05, high_threshold=5e-08,
                       title: None | str = None):
    """Display a Manhattan plot of the given DMR data, designed to work with the dataframe returned by get_dmrs()

    :param data_to_plot: (pd.DataFrame) dataframe to use for plotting.
    :param chromosome_col: (optional, string, default 'Chromosome') the name of the Chromosome column in the
        `data_to_plot` dataframe.
    :param x_col: (option, string, default 'Start') name of the column to use for X axis, start position of the probe/bin
    :param y_col: (optional, string, default 'p_value') the name of the value column in the `data_to_plot` dataframe
    :param annotation: (optional, Annotation, default None) Annotation data to use to annotation significant probes.
        Can be None to remove any annotation.
    :param annotation_col: (optional, str, default None) the name of a column used to write annotation on the
        plots for data that is above the significant threshold. Must be a column in the Annotation data
    :param medium_threshold: (optional, float, default 1e-05) set the threshold used for displaying annotation
        (and significance line if draw_significance is True)
    :param high_threshold: (optional, float, default 1e-08) set the threshold for the higher significance line (drawn if
        draw_significance is True)
    :param log10: (optional, boolean, default True) apply -log10 on the value column
    :param draw_significance: (option, boolean, default True) draw p-value significance lines (at 1e-05 and 5e-08)
    :param title: (optional, string) custom title for plot

    :return: nothing"""

    _manhattan_plot(data_to_plot=data_to_plot, chromosome_col=chromosome_col, y_col=y_col, x_col=x_col,
                    draw_significance=draw_significance, annotation=annotation, annotation_col=annotation_col,
                    medium_threshold=medium_threshold, high_threshold=high_threshold,
                    log10=log10, title=title)


def manhattan_plot_cnv(data_to_plot: pd.DataFrame, segments_to_plot=None, x_col='Start_bin',
                       chromosome_col='Chromosome',
                       y_col='cnv', title: None | str = None) -> None:
    """Display a Manhattan plot of the given CNV data, designed to work with the dataframes returned by
    copy_number_variation()

    :param data_to_plot: (pd.DataFrame) dataframe to use for plotting. Typically, the bins signal dataframe.
    :param segments_to_plot: (optional, pd.DataFrame) if set, display the segments using columns "chromosome", "start",
        "end" and "mean_cnv" of the given dataframe, where start and end are the position on the chromosome.
    :param chromosome_col: (optional, string, default 'Chromosome') the name of the Chromosome column in the
        `data_to_plot` dataframe.
    :param x_col: (option, string, default 'Start_bin') name of the column to use for X axis, start position of the
        probe/bin
    :param y_col: (optional, string, default 'cnv') the name of the value column in the `data_to_plot` dataframe
    :param title: (optional, string) custom title for plot

    :return: nothing"""

    _manhattan_plot(data_to_plot=data_to_plot, segments_to_plot=segments_to_plot, x_col=x_col,
                    chromosome_col=chromosome_col, y_col=y_col, title=title,
                    log10=False, annotation=None, draw_significance=False)