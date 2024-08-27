import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram

from illuminator.sample import Sample
from illuminator.samples import Samples
from illuminator.utils import get_chromosome_number


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
    manifest = sample.annotation.manifest
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
