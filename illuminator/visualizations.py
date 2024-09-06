import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

from illuminator.sample import Sample
from illuminator.samples import Samples
from illuminator.utils import get_chromosome_number, set_level_as_index

def plot_betas(betas: pd.DataFrame, n_bins: int = 100, title: None | str = None) -> None:
    """Plot betas values density for each sample

    :param betas: pd.DataFrame as output from sample[s].get_betas() - rows are probes and columns sample-s
    :param n_bins: number of bins to calculate histogram. Default to 100
    :param title: custom title for plot

    :return: None"""

    for index, row in betas.transpose().iterrows():
        histogram_values = np.histogram(row.dropna().values, bins=n_bins, density=True)
        plt.plot(histogram_values[1][:-1], histogram_values[0], label=index, linewidth=1)
    title = title if title is not None else f'Samples\' beta values distances on {len(betas)} probes'
    plt.title(title)
    plt.legend()


def betas_mds(betas: pd.DataFrame, colors: list | None = None, random_state: int = 42, title: None | str = None) -> None:
    """Plot samples in 2D space according to their beta distances.

    :param betas: pd.DataFrame as output from sample[s].get_betas() - rows are probes and columns sample-s
    :param colors: list of colors for each sample. Must have then length of N samples. Default to None
    :param random_state: seed for the MDS model. Assigning a seed makes the graphe reproducible across calls.
    :param title: custom title for plot

    :return: None"""

    # get betas with the most variance across samples (top 1000)
    betas_variance = np.var(betas, axis=1)
    indexes_most_variance = betas_variance.sort_values(ascending=False)[:1000].index
    betas_most_variance = betas.loc[indexes_most_variance].dropna()  # MDS doesn't support NAs

    # perform MDS
    mds = MDS(n_components=2, random_state=random_state)
    fit = mds.fit_transform(betas_most_variance.T)

    plt.scatter(x=fit[:, 0], y=fit[:, 1], label=betas.columns, c=colors)

    for index, name in enumerate(betas.columns):
        plt.annotate(name, (fit[index, 0], fit[index, 1]), fontsize=9)

    title = title if title is not None else f'Samples\' beta values distances on {len(betas)} probes'
    plt.title(title)


def betas_dendrogram(betas: pd.DataFrame, title: None | str = None) -> None:
    """Plot dendrogram of samples according to their beta values distances.

    :param betas: pd.DataFrame as output from sample[s].get_betas() - rows are probes and columns sample-s
    :param title: custom title for plot

    :return: None"""

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

    title = title if title is not None else f'Samples\' beta values distances on {len(betas)} probes'
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
        sns.heatmap(sorted_betas[:nb_probes].sort_values(betas.columns[0]))
    else:
        sns.clustermap(sorted_betas.dropna()[:nb_probes])

def _manhattan_plot(data_to_plot: pd.DataFrame, segments_to_plot: pd.DataFrame=None, chromosome_col='Chromosome',
                    x_col='Start', y_col='p_value', annotation_col=None, log10=False, title: None | str = None,
                    draw_significance=False) -> None:
    """Display a Manhattan plot of the given data.

    :param data_to_plot: (pd.DataFrame) dataframe to use for plotting. Typically, a dataframe returned by get_dmrs()
    :param segments_to_plot: (optional, pd.DataFrame) if set, display the segments using columns "chromosome", "start",
        "end" and "mean_cnv" of the given dataframe, where start and end are the position on the chromosome (as returned
        by copy_number_variation())
    :param chromosome_col: (optional, string, default 'Chromosome') the name of the Chromosome column in the
        `data_to_plot` dataframe.
    :param x_col: (option, string, default 'Start') name of the column to use for X axis, start position of the probe/bin
    :param y_col: (optional, string, default 'p_value') the name of the value column in the `data_to_plot` dataframe
    :param annotation_col: (optional, string, default 'probe_id') the name of a column used to write annotation on the
        plots for data that is above the significant threshold. Can be None to remove any annotation.
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

    # make indices for plotting
    data_to_plot_grouped = data_to_plot.groupby('chr_id', observed=True)

    # figure initialization
    fig, ax = plt.subplots(figsize=(14, 8))
    margin = int(max(data_to_plot[x_col]) / 10)
    chrom_start, chrom_end = 0, 0
    x_labels, x_major_ticks, x_minor_ticks = [], [], [0]
    high_threshold = 5e-08
    medium_threshold = 1e-05

    # apply -log10 to p-values if needed
    if log10:
        data_to_plot[y_col] = -np.log10(data_to_plot[y_col])
        high_threshold = -np.log10(high_threshold)
        medium_threshold = -np.log10(medium_threshold)

    # define colormap and limits
    v_max = np.max(data_to_plot[y_col])
    v_min = np.min(data_to_plot[y_col])
    if min(data_to_plot[y_col]) < 0:
        cmap = colormaps.get_cmap('gist_rainbow')
    else:
        cmap = colormaps.get_cmap('viridis').reversed()

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
            for chromosome in set(group[chromosome_col]):
                chrom_segments = segments_to_plot[segments_to_plot.chromosome == chromosome]
                for _, segment in chrom_segments.iterrows():
                    plt.plot([chrom_start + segment.start, chrom_start + segment.end],
                             [segment.mean_cnv, segment.mean_cnv],
                             c='black', linewidth=2)

        # draw annotations for probes that are over the threshold, if annotation_col is set
        if annotation_col is not None:
            indexes_to_annotate = group[y_col] > medium_threshold if log10 else group[y_col] < medium_threshold
            for _, row in group[indexes_to_annotate].iterrows():
                plt.annotate(row[annotation_col], (row[x_col] + 0.03, row[y_col] + 0.03), c=cmap(row[y_col]/v_max))

        chrom_start = chrom_end

    ax.set_facecolor('#EBEBEB')  # set background color to grey
    [ax.spines[side].set_visible(False) for side in ax.spines]  # hide plot frame

    # add lines of significance threshold
    if draw_significance:
        x_start = 0-margin
        x_end =  chrom_end + margin
        plt.plot([x_start, x_end], [high_threshold, high_threshold], linestyle='dotted', c=cmap(high_threshold), alpha=0.7)
        plt.plot([x_start, x_end], [medium_threshold, medium_threshold], linestyle='dotted', c=cmap(medium_threshold), alpha=0.5)

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
                   annotation_col='probe_id', log10=True, draw_significance=True, title: None | str = None):
    """Display a Manhattan plot of the given DMR data, designed to work with the dataframe returned by
    get_dmrs()

    :param data_to_plot: (pd.DataFrame) dataframe to use for plotting.
    :param chromosome_col: (optional, string, default 'Chromosome') the name of the Chromosome column in the
        `data_to_plot` dataframe.
    :param x_col: (option, string, default 'Start') name of the column to use for X axis, start position of the probe/bin
    :param y_col: (optional, string, default 'p_value') the name of the value column in the `data_to_plot` dataframe
    :param annotation_col: (optional, string, default 'probe_id') the name of a column used to write annotation on the
        plots for data that is above the significant threshold. Can be None to remove any annotation.
    :param log10: (optional, boolean, default True) apply -log10 on the value column
    :param draw_significance: (option, boolean, default True) draw p-value significance lines (at 1e-05 and 5e-08)
    :param title: (optional, string) custom title for plot

    :return: nothing"""

    _manhattan_plot(data_to_plot=data_to_plot, chromosome_col=chromosome_col, y_col=y_col, x_col=x_col,
                    draw_significance=draw_significance, annotation_col=annotation_col, log10=log10, title=title)

def manhattan_plot_cnv(data_to_plot: pd.DataFrame, segments_to_plot=None, x_col='Start_bin', chromosome_col='Chromosome',
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
                    log10=False, annotation_col=None, draw_significance=False)
