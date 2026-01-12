"""Functions to plot data from Samples object or DMPs/DMRs dataframes"""

import os.path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.patches import Patch
import matplotlib.text as mtext

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import linkage, dendrogram
from patsy import dmatrix
from statsmodels.api import OLS
import seaborn as sns
from KDEpy import FFTKDE

from pylluminator.samples import Samples
from pylluminator.annotations import Annotations
from pylluminator.dm import DM
from pylluminator.ml import dimensionality_reduction
from pylluminator.utils import get_chromosome_number, set_level_as_index, get_logger, merge_alt_chromosomes
from pylluminator.utils import merge_dataframe_by, merge_series_values


LOGGER = get_logger()

def _get_colors(sheet: pd.DataFrame, sample_label_name: str,
                color_column: str | list[str] | None, group_column: str | list[str] | None = None, 
                cmap_name: str = 'Spectral_r', na_color='darkgrey') -> tuple[list, dict | None]:
    """Define the colors to use for each sample, depending on the columns used to categorized them.

    :param sheet: sample sheet data frame
    :type sheet: pandas.DataFrame

    :param sample_label_name: the name of the sample sheet column used as label
    :type sample_label_name: str

    :param color_column: name of the column of the sample sheet to use for color. If None, the function will return empty objects.
    :type color_column: str | None

    :param group_column: name of the column of the sample sheet to use for grouping samples. If not None, samples from the same group will have the same color.
    :type group_column: str | list[str] | None

    :param cmap_name: name of the matplotlib color map to use. Default: Spectral_r
    :type cmap_name: str

    :param na_color: color for NA values. Default: darkgrey
    :type na_color: str

    :return: the list of legend handles and a dict of color categories, with keys being the values of color_column
    :rtype: tuple[list,dict | None]"""

    # check input parameters (the columns must exist in the sample sheet)
        
    if group_column is not None:
        group_column = group_column if isinstance(group_column, list) else [group_column]
        for col in group_column:
            if col is not None and col not in sheet.columns:
                LOGGER.warning(f'Group column {col} not found in the sample sheet, ignoring it')
        group_column = [col for col in group_column if col in sheet.columns]
        if len(group_column) == 0:
            group_column = None
    
    if color_column is not None:
        color_column = color_column if isinstance(color_column, list) else [color_column]
        for col in color_column:
            if col is not None and col not in sheet.columns:
                LOGGER.warning(f'Color column {col} not found in the sample sheet, ignoring it')
        color_column = [col for col in color_column if col in sheet.columns]
        if len(color_column) == 0:
            color_column = None
        
    if color_column is None and group_column is None:
        return [], None

    legend_handles = []
    color_categories = dict()
    cmap = colormaps[cmap_name]

    # if there is one sample per color, avoid creating one category per sample
    if group_column is None and sample_label_name is not None and color_column == [sample_label_name]:
        color_categories = {name: cmap(i / len(sheet)) for i, name in enumerate(sheet[sample_label_name])}
        legend_handles += [Line2D([0], [0], color=color, label=label) for label, color in color_categories.items()]
    else:
        groupby_cols = group_column if group_column is not None else color_column
        grouped_sheet = sheet.groupby(groupby_cols, dropna=False, observed=True)

        def get_group_color(group, value_idx: int | float):
            # more than one column, or non numeric column
            if len(groupby_cols) > 1 or not is_numeric_dtype(sheet[groupby_cols[0]]):
                nb_colors = max(1, grouped_sheet.ngroups - 1)
                return cmap(value_idx / nb_colors)
            # if one numeric column, get proportional colors
            min_val = sheet[groupby_cols[0]].min()
            max_val = sheet[groupby_cols[0]].max()
            value = group[groupby_cols[0]].values[0]            
            if min_val == max_val:
                return cmap(0)
            return cmap((value - min_val) / (max_val - min_val))            
        
        if group_column is None:
            legend_handles += [Line2D([0], [0], color='black', linestyle='', label=", ".join(color_column))]
        
        for i, (group_name, group) in enumerate(grouped_sheet):
            group_name = list(group_name)  # convert tuple to list
            color = na_color if pd.isna(group_name).all() else get_group_color(group, i)
            group_name = ', '.join([str(gn) for gn in group_name])#.replace("'", "").replace('(','').replace(')','')
            if group_column is None:
                for name in group[sample_label_name]:
                    color_categories[name] = color
                legend_handles += [mpatches.Patch(color=color, label=group_name)]
            else:
                color_categories[group_name] = color
                
    return legend_handles, color_categories


def _get_linestyles(sheet: pd.DataFrame, column: str | None) -> tuple[list, dict | None]:
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
    line_styles = ['solid', 'dotted', 'dashed', 'dashdot']

    # determine the line style
    categories = sorted(set(sheet[column]))
    for i, category in enumerate(categories):
        linestyle_categories[category] = line_styles[i % len(line_styles)]
    legend_handles += [Line2D([0], [0], color='black', linestyle='', label=f'{column} :')]
    legend_handles += [Line2D([0], [0], color='black', linestyle=ls, label=label) for label, ls in
                       linestyle_categories.items()]

    return legend_handles, linestyle_categories


def betas_density(samples: Samples, title: None | str = None, group_column: None | str | list[str] = None,
               color_column: str | None = None,  linestyle_column=None, figsize=(10, 7), cmap_name: str = 'Spectral_r',
               custom_sheet: None | pd.DataFrame = None, apply_mask=True, alpha=0.8, save_path: None | str=None) -> None:
    """Plot beta values density for each sample

    :param samples: with beta values already calculated
    :type samples: Samples

    :param title: custom title for the plot to override generated title. Default: None
    :type title: str | None

    :param color_column: name of a Sample Sheet column to define which samples get the same color. Default: None
    :type color_column: str | None

    :param group_column: compute the average beta values per group of samples. Default: None
    :type group_column: str | list[str] | None

    :param linestyle_column: name of a Sample Sheet column to define which samples get the same line style. Default: None
    :type linestyle_column: str | None

    :param figsize: size of the figure. Default: (10, 7)
    :type figsize: tuple

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the samples to display
    :type custom_sheet: pandas.DataFrame | None

    :param apply_mask: true removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param cmap_name: name of the matplotlib color map to use. Default: Spectral_r
    :type cmap_name: str

    :param alpha: transparence of the lines. Float between 0 and 1. Default: 0.8
    :type alpha: float
    
    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str

    :return: None"""

    # initialize values
    plt.style.use('ggplot')
    plt.subplots(figsize=figsize)

    # get betas with or without masked probes and samples
    betas = samples.get_betas(apply_mask=apply_mask, custom_sheet=custom_sheet, drop_na=True)
    if betas is None or len(betas) == 0:
        LOGGER.error('No betas to plot')
        return

    sheet = samples.sample_sheet[samples.sample_sheet[samples.sample_label_name].isin(betas.columns)]

    if group_column is not None:

        grouped_sheet = sheet.groupby(group_column, observed=True)
        avg_betas_list = []
        group_names = []
        for name, line in grouped_sheet[samples.sample_label_name].apply(list).items():
            avg_betas_list.append(betas[line].mean(axis=1))
            group_names.append(str(name))

        betas = pd.concat(avg_betas_list, axis=1)
        betas.columns = group_names

    # define the color and line style of each sample
    c_legend_handles, colors = _get_colors(sheet, samples.sample_label_name, color_column, group_column, cmap_name=cmap_name)
    ls_legend_handles, linestyles = _get_linestyles(sheet, linestyle_column)
    legend_handles = c_legend_handles + ls_legend_handles

    for c in betas.columns:
        x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(betas[c].values).evaluate()
        ls = '-' if linestyles is None else linestyles[c]
        col = None if colors is None else colors[c]
        plt.plot(x, y, label=c, color=col, linestyle=ls, alpha=alpha)

    if title is None:
        if group_column is None:
            title = f'Beta values of {len(betas.columns)} samples on {len(betas):,} probes'
        else:
            title = f'Beta values of {len(betas.columns)} groups of samples on {len(betas):,} probes'
        if apply_mask:
            title += ' (masks applied)'

    plt.title(title)

    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))


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


def betas_2D(samples: Samples, label_column: str | None=None, color_column: str | list[str] | None=None,dims_to_plot:list[int]=[1, 2],
             nb_probes: int | None=None, title: None | str = None, apply_mask=True, figsize: tuple[float, float]|None=(10, 7),
             cmap_name: str = 'Spectral_r', na_color: str|None='darkgrey',
             custom_sheet: None | pd.DataFrame = None, save_path: None | str=None, model='PCA', show_labels=True, plot_kwargs: dict|None=None, **kwargs) -> None:
    """Plot samples in 2D space according to their beta distances.

    :param samples : samples to plot
    :type samples: Samples

    :param label_column: name of the column containing the labels. Default: None
    :type label_column: str | None

    :param color_column: name of a Sample Sheet column used to give samples from the same group the same color. Default: None
    :type color_column: str | list[str]| None

    :param nb_probes: number of probes to use for the model, selected from the probes with the most beta variance.
        If None, use all the probes. Default: None
    :type nb_probes: int | None

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param figsize: size of the plot. Default: (10, 7)
    :type figsize: tuple

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :param model: identifier of the model to use. Available models are 'PCA': PCA, 'MDS': MDS, 'DL': DictionaryLearning,
        'FA': FactorAnalysis, 'FICA': FastICA, 'IPCA': IncrementalPCA, 'KPCA': KernelPCA, 'LDA': LatentDirichletAllocation,
        'MBDL': MiniBatchDictionaryLearning, 'MBNMF': MiniBatchNMF, 'MBSPCA': MiniBatchSparsePCA, 'NMF': NMF,
        'SPCA': SparsePCA, 'TSVD': TruncatedSVD. Default: 'PCA'
    :type model: str

    :param cmap_name: name of the matplotlib color map to use. Default: Spectral_r
    :type cmap_name: str

    :param na_color: color for NaN values
    :type na_color: str

    :param show_labels: if True, show the sample names on the plot. Default: True
    :type show_labels: bool

    :param kwargs: parameters passed to the model

    :return: None"""

    # check input parameters
    if label_column is None:
        label_column = samples.sample_label_name
    if label_column not in samples.sample_sheet.columns:
        LOGGER.warning(f'Label column {label_column} not found in the sample sheet, setting it to default')
        label_column = samples.sample_label_name

    if color_column is None:
        color_column = samples.sample_label_name

    if 'n_components' not in kwargs:
        kwargs['n_components'] = max(dims_to_plot)
    elif kwargs['n_components'] < max(dims_to_plot):
        LOGGER.warning(f'n_components is set to {kwargs["n_components"]}, but dims_to_plot contains {max(dims_to_plot)}. Setting n_components to {max(dims_to_plot)}')
        kwargs['n_components'] = max(dims_to_plot)
    if 'random_state' not in kwargs and model not in ['IPCA']:
        kwargs['random_state'] = 42

    fitted_model, reduced_data, labels, nb_probes_used = dimensionality_reduction(samples, model=model, nb_probes=nb_probes,
                                                     custom_sheet=custom_sheet, apply_mask=apply_mask, **kwargs)
    if fitted_model is None:
        return

    sheet = custom_sheet if custom_sheet is not None else samples.sample_sheet
    legend_handles, colors_dict = _get_colors(sheet, label_column, color_column, cmap_name=cmap_name, na_color=na_color)
    if label_column != samples.sample_label_name:
        labels = [sheet[sheet[samples.sample_label_name] == label][label_column].values[0] for label in labels]

    plt.style.use('ggplot')
    plt.figure(figsize=figsize)
    if plot_kwargs is None:
        plot_kwargs =  {}
    colors = [colors_dict[label] for label in labels] if colors_dict is not None else None
    plt.scatter(x=reduced_data[:, dims_to_plot[0]-1], y=reduced_data[:, dims_to_plot[1]-1], label=labels, c=colors, **plot_kwargs)

    if model in ['PCA', 'ICPA', 'TSVD']:
    # if hasattr(model_ini, 'explained_variance_ratio_'):
        plt.xlabel(f'component {dims_to_plot[0]}:{fitted_model.explained_variance_ratio_[dims_to_plot[0]-1]*100:.2f}%' )
        plt.ylabel(f'component {dims_to_plot[1]}:{fitted_model.explained_variance_ratio_[dims_to_plot[1]-1]*100:.2f}%' )

    if show_labels:
        for index, name in enumerate(labels):
            plt.annotate(name, (reduced_data[index, dims_to_plot[0]-1], reduced_data[index, dims_to_plot[1]-1]), fontsize=9)

    if title is None:
        if nb_probes is None:
            title = f'{model} of all the probes ({nb_probes_used:,})'
        else:
            title = f'{model} of the {nb_probes_used:,} most variable probes'
        if apply_mask:
            title += ' (masks applied)'

    plt.title(title)

    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

def _pc_heatmap(samples: Samples, type:str, params: list[str] | None = None, nb_probes: int | None = None,
                apply_mask=True, vmax=0.05, custom_sheet: None | pd.DataFrame = None,
                abs_corr=None, sig_threshold=None,
                save_path: None | str = None, model='PCA',  orientation='v', **kwargs):
    """ Heatmap of the p-values for the association of principal components and the parameters in the sample sheet.

    :param samples: samples to plot
    :type samples: Samples

    :param params: list of parameters to correlate with the principal components. Must be columns of the sample sheet.
        If None, show all parameters. Default: None
    :type params: list[str] | None

    :param type: 'association' for p-values of the association, 'correlation' for correlation values
    :type type: str

    :param nb_probes: number of probes to use for the model, selected from the probes with the most beta variance.
        If None, use all the probes. Default: None
    :type nb_probes: int | None

    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param vmax: maximum value for the color scale. Default: 0.05
    :type vmax: float

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :param model: identifier of the model to use. Available models are 'PCA': PCA, 'MDS': MDS, 'DL': DictionaryLearning,
        'FA': FactorAnalysis, 'FICA': FastICA, 'IPCA': IncrementalPCA, 'KPCA': KernelPCA, 'LDA': LatentDirichletAllocation,
        'MBDL': MiniBatchDictionaryLearning, 'MBNMF': MiniBatchNMF, 'MBSPCA': MiniBatchSparsePCA, 'NMF': NMF,
        'SPCA': SparsePCA, 'TSVD': TruncatedSVD. Default: 'PCA'
    :type model: str

    :param orientation: orientation of the heatmap. Possible values: 'v', 'h'. Default: 'v'
    :type orientation: str

    :param kwargs: parameters passed to the model

    :return: None
    """

    if type not in ['association', 'correlation']:
        LOGGER.error(f'Unknown type {type}. Known types are "association" and "correlation"')
        return
    
    # fit the model
    fitted_model, reduced_data, labels, nb_probes = dimensionality_reduction(samples, model=model, nb_probes=nb_probes,
                                                     custom_sheet=custom_sheet, apply_mask=apply_mask, **kwargs)
    if fitted_model is None:
        return

    sheet = custom_sheet if custom_sheet is not None else samples.sample_sheet
    sample_info = sheet.set_index(samples.sample_label_name).loc[labels]
    # drop columns with only NaN values that cant be used in the model
    sample_info = sample_info.dropna(axis=1, how='all')

    if len(sample_info) == 0:
        LOGGER.error('No sample to plot')
        return
    
    result = pd.DataFrame(dtype=float)
    skipped_params = []
    n_components = min(20, reduced_data.shape[1])

    # no specific parameter defined, show them all (except the samples identifiers)
    if params is None:
        params = sheet.columns.to_list()
        params.remove(samples.sample_label_name)

    for param in params:

        if param not in sample_info.columns:
            LOGGER.warning(f'Parameter {param} not found in the sample sheet, skipping')            
            continue

        # skip parameters with only one value
        param_values = set(sample_info[param].dropna())
        if len(param_values) == 1 or len(param_values) == len(sample_info):
            skipped_params.append(param)
            continue

        # the design matrix removes the NaN values
        design_matrix = dmatrix(f'~ {param}', sample_info, return_type='dataframe')
        # remove columns that contain only 0 (due to absent categories e.g)
        design_matrix = design_matrix[[c for c in design_matrix.columns if ~(design_matrix[c].values == 0).all()]]
        if design_matrix.empty:
            skipped_params.append(param)
            continue

        for i in range(n_components):
            # stop when components explain less than 1% of the variance
            if fitted_model.explained_variance_ratio_[i] < 0.01:
                break
            fitted_ols = OLS(reduced_data[~sample_info[param].isna(), i], design_matrix, missing='drop').fit()
            if type == 'correlation':
                if sig_threshold is not None and fitted_ols.f_pvalue > sig_threshold:
                    continue
                factor = -1 if not abs_corr and fitted_ols.params.iloc[1] < 0 else 1  # direction of the correlation
                result.loc[i, param] = factor * np.sqrt(fitted_ols.rsquared)  # correlation
            elif type == 'association':
                result.loc[i, param] = fitted_ols.f_pvalue
            result.loc[i, 'principal component'] = f'{int(i+1)} ({fitted_model.explained_variance_ratio_[i]*100:.2f}%)'

    if len(skipped_params) > 0:
        LOGGER.warning(f'Parameters {", ".join(skipped_params)} have no effect')

    if result.empty:
        if type == 'correlation':
            LOGGER.warning('No significant correlation found')
        else:
            LOGGER.warning('No significant component')
        return
    
    result = result.sort_index()

    # drop columns with only NA values (eg Sample Ids that are unique per sample)
    result = result.set_index('principal component').dropna(axis=1, how='all')

    if orientation == 'v':
        result = result.T

    # finally plot the results
    figsize = (max(4, len(result.columns)), max(4, len(result)/2))
    cmap = sns.cm.rocket_r if type == 'association' else sns.cm.rocket_r if abs_corr else 'vlag'
    vmin = 0 if abs_corr or type == 'association' else -1
    fmt = '.2f' if type == 'correlation' else '.0e'
    plt.subplots(figsize=figsize, constrained_layout=True)
    ax = sns.heatmap(result, annot=True, fmt=fmt, vmax=vmax, vmin=vmin, cmap=cmap)
    ax.tick_params(left=False, bottom=False, labelbottom=False, labeltop=True)
    plt.xticks(rotation=45, ha='left')
    if save_path is not None:
        ax.get_figure().savefig(os.path.expanduser(save_path))


def pc_association_heatmap(samples: Samples, params: list[str] | None = None, nb_probes: int | None = None,
                           apply_mask=True, vmax=0.05, custom_sheet: None | pd.DataFrame = None,
                           save_path: None | str = None, model='PCA',  orientation='v', **kwargs):
    """ Heatmap of the p-values for the association of principal components and the parameters in the sample sheet.

    :param samples: samples to plot
    :type samples: Samples

    :param params: list of parameters to correlate with the principal components. Must be columns of the sample sheet.
        If None, show all parameters. Default: None
    :type params: list[str] | None

    :param nb_probes: number of probes to use for the model, selected from the probes with the most beta variance.
        If None, use all the probes. Default: None
    :type nb_probes: int | None

    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param vmax: maximum value for the color scale. Default: 0.05
    :type vmax: float

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :param model: identifier of the model to use. Available models are 'PCA': PCA, 'MDS': MDS, 'DL': DictionaryLearning,
        'FA': FactorAnalysis, 'FICA': FastICA, 'IPCA': IncrementalPCA, 'KPCA': KernelPCA, 'LDA': LatentDirichletAllocation,
        'MBDL': MiniBatchDictionaryLearning, 'MBNMF': MiniBatchNMF, 'MBSPCA': MiniBatchSparsePCA, 'NMF': NMF,
        'SPCA': SparsePCA, 'TSVD': TruncatedSVD. Default: 'PCA'
    :type model: str

    :param orientation: orientation of the heatmap. Possible values: 'v', 'h'. Default: 'v'
    :type orientation: str

    :param kwargs: parameters passed to the model

    :return: None
    """
    _pc_heatmap(samples, 'association', params=params, nb_probes=nb_probes, apply_mask=apply_mask, vmax=vmax,
                custom_sheet=custom_sheet, save_path=save_path, model=model, orientation=orientation, **kwargs)


def pc_correlation_heatmap(samples: Samples, params: list[str] | None = None, nb_probes: int | None = None,
                           apply_mask=True, custom_sheet: None | pd.DataFrame = None, abs_corr=True, sig_threshold:float|None=0.05,
                           save_path: None | str = None, model='PCA',  orientation='v', **kwargs):
    """ Heatmap of the correlation between principal components and the metadata of the sample sheet.

    :param samples: samples to plot
    :type samples: Samples

    :param params: list of parameters to correlate with the principal components. Must be columns of the sample sheet.
        If None, show all parameters. Default: None
    :type params: list[str] | None

    :param nb_probes: number of probes to use for the model, selected from the probes with the most beta variance.
        If None, use all the probes. Default: None
    :type nb_probes: int | None

    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param abs_corr: if True, show the absolute value of the correlation. Default: True
    :type abs_corr: bool

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :param model: identifier of the model to use. Available models are 'PCA': PCA, 'MDS': MDS, 'DL': DictionaryLearning,
        'FA': FactorAnalysis, 'FICA': FastICA, 'IPCA': IncrementalPCA, 'KPCA': KernelPCA, 'LDA': LatentDirichletAllocation,
        'MBDL': MiniBatchDictionaryLearning, 'MBNMF': MiniBatchNMF, 'MBSPCA': MiniBatchSparsePCA, 'NMF': NMF,
        'SPCA': SparsePCA, 'TSVD': TruncatedSVD. Default: 'PCA'
    :type model: str

    :param orientation: orientation of the heatmap. Possible values: 'v', 'h'. Default: 'v'
    :type orientation: str

    :param kwargs: parameters passed to the model

    :return: None
    """

    _pc_heatmap(samples, 'correlation', params=params, nb_probes=nb_probes, apply_mask=apply_mask, custom_sheet=custom_sheet,
                abs_corr=abs_corr, sig_threshold=sig_threshold, save_path=save_path, model=model, orientation=orientation, vmax=1, **kwargs) 


def betas_dendrogram(samples: Samples, title: None | str = None, color_column: str|None=None,
                     figsize:tuple[float, float]=(10, 7), cmap_name: str = 'Spectral_r',
                     custom_sheet: pd.DataFrame | None = None, apply_mask: bool = True, save_path: None | str=None) -> None:
    """Plot dendrogram of samples according to their beta values distances.

    :param samples: samples to plot
    :type samples: Samples

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param color_column: name of a Sample Sheet column used to give samples from the same group the same color. Default: None
    :type color_column: str

    :param figsize: size of the plot. Default: (10, 7)
    :type figsize: tuple

    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param cmap_name: name of the matplotlib color map to use. Default: Spectral_r
    :type cmap_name: str

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""
    plt.style.use('ggplot')
    plt.figure(figsize=figsize)

    betas = samples.get_betas(drop_na=True, apply_mask=apply_mask, custom_sheet=custom_sheet)
    if betas is None or len(betas) == 0:
        LOGGER.error('No betas to plot')
        return

    sheet = samples.sample_sheet[samples.sample_sheet[samples.sample_label_name].isin(betas.columns)]

    linkage_matrix = linkage(betas.T.values, optimal_ordering=True, method='complete')
    dendrogram(linkage_matrix, labels=betas.columns, orientation='left')

    if color_column is not None:
        legend_handles, label_colors = _get_colors(sheet, samples.sample_label_name, color_column=color_column, cmap_name=cmap_name)

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

    if title is None:
        title = f'Samples\' beta values distances on {len(betas):,} probes'
        if apply_mask:
            title += ' (masks applied)'
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))


########################################################################################################################


def nb_probes_per_chr_and_type_hist(samples: Samples, title: None | str = None, figsize:tuple[float, float]=(10, 7), save_path: None | str=None) -> None:
    """Plot the number of probes covered by the sample per chromosome and design type

    :param sample: Samples to be plotted
    :type sample: Samples

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :param figsize: size of the plot. Default: (10, 7)
    :type figsize: tuple

    :return: None"""

    chromosome_df, type_df = samples.get_nb_probes_per_chr_and_type()

    plt.style.use('ggplot')
    fig, axes = plt.subplots(2)

    # make 'masked' bar blue and 'not masked' orange
    chromosome_df.plot.bar(stacked=True, figsize=figsize, ax=axes[0], color=['#1f77b4', '#F8766D'] )
    type_df.plot.bar(stacked=True, figsize=figsize, ax=axes[1], color=[ '#1f77b4', '#F8766D',] )

    if title is None:
        title = f'Number of probes per chromosome and type for {samples.nb_samples} samples'

    fig.suptitle(title)

    # rotate x labels for better readability
    plt.setp(axes[0].get_xticklabels(), rotation=0, ha='center')
    plt.setp(axes[1].get_xticklabels(), rotation=0, ha='center')

    # thousand separator for y axis
    axes[0].yaxis.set_major_formatter('{x:,.0f}')
    axes[1].yaxis.set_major_formatter('{x:,.0f}')
    axes[0].set_ylabel('Number of probes')
    axes[1].set_ylabel('Number of probes')

    #add space between plots
    plt.subplots_adjust(hspace=0.3)

    if save_path is not None:
        fig.savefig(os.path.expanduser(save_path))

########################################################################################################################

class _LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(_LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        handlebox.add_artist(title)
        return title

def _convert_df_values_to_colors(input_df: pd.DataFrame, legend_names: list[str] | None = None) -> tuple:
    """Treat each column of the dataframe as a distinct category, and convert its values to colors. If the values are
    string, treat them as categories, if they are numbers, use a continuous colormap. Generate the associated legend
    handles for the specified names.

    :param input_df: dataframe
    :type input_df: pandas.DataFrame

    :param legend_names: list of columns to generate legend handles for. If None or empty, don't generate any. Default: None
    :type legend_names: list[str] | None

    :return: the colors dataframe and the legend handles as a list of tuples (legend name, df[value name, corresponding color])
    :rtype: tuple(pandas.DataFrame, list)
    """
    string_cmap_index = 0
    number_cmap_index = 0
    string_cmaps = ['hls', 'pastel', 'dark']
    number_cmaps = ['viridis', 'plasma', 'cool', 'spring']

    def get_string_color(val, nb_cats):
        if pd.isna(val):
            return (1, 1, 1, 1)
        return sns.color_palette(string_cmaps[string_cmap_index % len(string_cmaps)], int(nb_cats))[int(val)]

    def get_numeric_color(val, cmin, cmax):
        if pd.isna(val):
            return (1, 1, 1, 1)
        norm = plt.Normalize(cmin, cmax)
        return colormaps.get_cmap(number_cmaps[number_cmap_index % len(number_cmaps)])(norm(val))

    how_to = {column: merge_series_values for column in input_df.columns}
    if input_df.index.name is None:
        input_df.index.name = 'index'
    input_df = input_df.groupby(input_df.index.name).agg(how_to)
    color_df = input_df.copy()

    handles = []
    labels = []
    colormaps_list = []

    for col in input_df.columns:
        # get colors
        if input_df[col].dtype in ['object', 'category']:
            # convert string category codes to easily get a color index for each string
            color_df[col] = pd.Categorical(input_df[col]).codes
            color_df.loc[color_df[col] == -1, col] = None
            color_df[col] = color_df[col].apply(get_string_color, args=(color_df[col].max()+1,))
            string_cmap_index += 1
        elif np.issubdtype(input_df[col].dtype, np.number):
            color_df[col] = input_df[col].apply(get_numeric_color, args=(input_df[col].min(), input_df[col].max()))
            number_cmap_index += 1
        # make legends (category title + colors & labels)
        if legend_names is not None and col in legend_names:
            legend_df = pd.concat([color_df[col], input_df[col]], axis=1).drop_duplicates()
            legend_df.columns = ['color', 'name']
            legend_df = legend_df.sort_values('name')
            if input_df[col].dtype in ['object', 'category']:
                handles += [col] + legend_df.color.apply(lambda x: Patch(color=x)).tolist()
                labels += [''] + legend_df.name.values.tolist()
            else:
                # for numeric columns, create a colorbar instead of individual patches
                sm = plt.cm.ScalarMappable(cmap=number_cmaps[(number_cmap_index-1) % len(number_cmaps)],
                                           norm=plt.Normalize(vmin=input_df[col].min(), vmax=input_df[col].max()))
                sm.set_array([])
                colormaps_list.append([col, sm])

    return color_df, handles, labels, colormaps_list


def betas_heatmap(samples: Samples, apply_mask:bool=True,
                     nb_probes: int = 100, figsize: tuple[float, float] = (10, 10),
                     var: str | None | list[str] = None, custom_sheet: pd.DataFrame | None = None,
                     drop_na=True, save_path: None | str = None,
                     row_factors: str | list[str] | None = None, row_legends: str | list[str] | None = '') -> None:
    """Plot a heatmap of the probes with the most variable beta values, showing hierarchical clustering of the probes
    with dendrograms on the sides.

    :param samples: samples to use for plotting
    :type samples: Samples
    :param nb_probes: number of probes to plot. Default: 100
    :type nb_probes: int
    :param figsize: size of the plot. Default: (10, 10)
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
    :param row_factors: list of columns to show as color categories on the side of the heatmap. Must correspond to
        columns of the sample sheet. Default: None
    :type row_factors: str | list[str] | None
    :param row_legends: list of columns to generate a legend for. Set to None for no legends. Only work for columns also
        specified in row_factors. Default: '' (generate all legends)
    :type row_legends: str | list[str] | None

    :return: None"""


    label = samples.sample_label_name
    betas = samples.get_betas(custom_sheet=custom_sheet, drop_na=drop_na, apply_mask=apply_mask)

    if betas is None or len(betas) == 0:
        LOGGER.error('No beta values found')
        return None

    betas = set_level_as_index(betas, 'probe_id', drop_others=True)

    # get betas with the most variance across samples
    betas_variance = np.var(betas, axis=1)
    nb_probes = min(nb_probes, len(betas_variance))
    indexes_most_variance = betas_variance.sort_values(ascending=False)[:nb_probes].index
    betas = betas.loc[indexes_most_variance]

    sheet = custom_sheet if custom_sheet is not None else samples.sample_sheet
    sheet = sheet.copy()[sheet[label].isin(betas.columns)]  # get intersection of the two dfs
    sheet = sheet.set_index(label)

    # add values next to sample labels if var is specified and use the new labels as betas column names
    if var is not None:
        if isinstance(var, str):
            var = [var]
        # update beta column names and sheet labels by adding values of 'var' columns after the sample names
        new_labels = [f'{c} ({",".join([str(sheet.loc[c, v]) for v in var])})' for c in betas.columns]
        sheet = sheet.loc[betas.columns,]  # sort sheet like beta columns
        sheet.index = new_labels  # update sheet index values
        sheet.index.name = samples.sample_label_name
        betas.columns = new_labels

    # common parameters to clustermap and heatmap
    colormaps_list = []
    heatmap_params = {'yticklabels': True, 'xticklabels': True, 'cmap': 'Spectral_r', 'vmin': 0, 'vmax': 1}
    legend_params = {'handler_map': {str: _LegendTitle({'fontweight': 'bold'})}, 'loc': 'upper right',
                     'bbox_to_anchor': (0, 1)}
    betas = betas.T

    if drop_na:
        handles, labels = [], []
        # convert categories to colors and get legends if specified
        if row_factors is not None:
            row_factors = [row_factors] if isinstance(row_factors, str) else row_factors
            if row_legends == '':
                row_legends = row_factors
            elif isinstance(row_legends, str):
                row_legends = [row_legends]
            subset = sheet[row_factors]
            row_factors, handles, labels, colormaps_list = _convert_df_values_to_colors(subset, row_legends)

        # plot the heatmap
        plot = sns.clustermap(betas, row_colors=row_factors, figsize=figsize, **heatmap_params)
        # reshape colorbar to be horizontal
        x0, _, _, _ = plot.cbar_pos
        plot.ax_cbar.set_position([x0, 0.9, plot.ax_row_dendrogram.get_position().width, 0.02])
        # add the legends if they exist
        leg_margin = 0
        if len(handles) > 0 and len(labels) > 0:
            plt.legend(handles=handles, labels=labels, **legend_params)
            leg_margin = -0.15

        if len(colormaps_list) > 0:
            fig = plot.fig       
            for i, (label, sm) in enumerate(colormaps_list):
                cbar_ax = fig.add_axes([-0.1+leg_margin, 0.9-(i+1)*0.23, 0.02, 0.2])  # [left, bottom, width, height]
                fig.colorbar(sm, cax=cbar_ax, label=label)

        # save plot
        if save_path is not None:
            plot.savefig(os.path.expanduser(save_path))
    else:
        if row_factors is not None:
            LOGGER.warning(f'Parameter {row_factors} is ignored when drop_na is False')

        plot = sns.heatmap(betas, **heatmap_params)

        if save_path is not None:
            plot.get_figure().savefig(os.path.expanduser(save_path))



def dmp_heatmap(dm: DM, contrast: str | None = None,
                     nb_probes: int = 100, figsize: tuple[float, float] | None = None,
                     var: str | None | list[str] = None, custom_sheet: pd.DataFrame | None = None,
                     drop_na=True, save_path: None | str = None,
                     sort_by = 'effect_size', ascending = False,
                     pval_threshold: float | None = 0.05, effect_size_threshold: float | None = None,
                     row_factors: str | list[str] | None = None, row_legends: str | list[str] | None = '', xticklabels=True) -> None:
    """Plot a heatmap of the probes that are the most differentially methylated, showing hierarchical clustering of the
    probes with dendrograms on the sides.


    :param contrast: name of the contrast to use to sort beta values. Must be one of the contrasts of the DM object.
        If None is given, will use the F-statistics p-value. Default: None
    :type contrast: str | None

    :param nb_probes: number of probes to plot. Default: 100
    :type nb_probes: int

    :param figsize: size of the plot. Default: None (automatic)
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

    :param sort_by: column to order dmps by. Can be pvalue, effect_size, or any dmp column name. Default: effect_size
    :type sort_by: str

    :param ascending: sorting order. Default: False
    :type ascending: bool

    :param pval_threshold: maximum DMPs p-value, float between 0 and 1. If None, no filter is applied on pvalues. Default: 0.05
    :type pval_threshold: float | None

    :param effect_size_threshold: minimum DMPs effect size, float between 0 and 1. If None, no filter is applied on effect size. Default: None
    :type effect_size_threshold: float | None

    :param row_factors: list of columns to show as color categories on the side of the heatmap. Must correspond to
        columns of the sample sheet. Default: None
    :type row_factors: str | list[str] | None

    :param row_legends: list of columns to generate a legend for. Set to None for no legends. Only work for columns also
        specified in row_factors. Default: '' (generate all legends)
    :type row_legends: str | list[str] | None

    :return: None"""

    if dm is None or dm.dmp is None or len(dm.dmp) == 0:
        LOGGER.error('No DMPs to plot')
        return

    if isinstance(contrast, list):
        LOGGER.error('dmp_heatmap() : contrast must be a string, not a list')
        return

    # no F-statistic p_value for DMPs calculated with MixedModelLM, so we need a contrast specified
    if pd.isna(dm.dmp['f_pvalue']).all() and contrast is None:
        LOGGER.error('You need to specify a contrast for DMPs calculated with a mixed model')
        return None

    if sort_by not in ['pvalue'] + dm.dmp.columns.tolist():
        LOGGER.error(f'parameter sort_by={sort_by} not found. Must be "pvalue" or any column of dmps dataframe')
        return

    label = dm.samples.sample_label_name
    betas = dm.samples.get_betas(custom_sheet=custom_sheet, drop_na=drop_na)
    if betas is None or len(betas) == 0:
        LOGGER.error('No beta values found')
        return None

    betas = set_level_as_index(betas, 'probe_id', drop_others=True)

    # filter DMPs by p-value threshold and effect size threshold if the parameters are set
    dmp = dm.dmp
    pval_column = 'f_pvalue' if contrast is None else f'{contrast}_p_value_adjusted'

    if pval_threshold is not None:
        dmp = dmp[dmp[pval_column] <= pval_threshold]

    if effect_size_threshold is not None:
        dmp = dmp[dmp['effect_size'] >= effect_size_threshold]

    # sort the DMPs
    if sort_by == 'pvalue':
        sort_by = pval_column
    sorted_probe_idxs = dmp.sort_values(sort_by, ascending=ascending).index.intersection(betas.index)

    nb_probes = min(nb_probes, len(sorted_probe_idxs))

    if nb_probes == 0:
        LOGGER.error(f'No significant probes found, consider increasing the p-value threshold (current threshold: {pval_threshold}) or decreasing the effect size threshold (current threshold: {effect_size_threshold}))')
        return

    sorted_probe_idxs = sorted_probe_idxs[:nb_probes]

    if betas is None or len(betas) == 0:
        LOGGER.error('No betas to plot')
        return

    sheet = custom_sheet if custom_sheet is not None else dm.sample_info
    sheet = sheet.copy()[sheet[label].isin(betas.columns)]  #  get intersection of the two dfs
    sheet = sheet.set_index(label)

    # add values next to sample labels if var is specified and use the new labels as betas column names
    if var is not None:
        if isinstance(var, str):
            var = [var]
        # update beta column names and sheet labels by adding values of 'var' columns after the sample names
        new_labels = [f'{c} ({",".join([str(sheet.loc[c, v]) for v in var])})' for c in betas.columns]
        sheet = sheet.loc[betas.columns,]  # sort sheet like beta columns
        sheet.index = new_labels  # update sheet index values
        sheet.index.name = dm.samples.sample_label_name
        betas.columns = new_labels

    betas = betas.loc[sorted_probe_idxs].T
    
    # common parameters to clustermap and heatmap
    heatmap_params = {'yticklabels': True, 'xticklabels': xticklabels, 'cmap': 'Spectral_r', 'vmin': 0, 'vmax': 1, 'cbar_kws': {'label': 'Beta value', 'orientation': 'horizontal'}}
    legend_params = {'handler_map': {str: _LegendTitle({'fontweight': 'bold'})}, 'loc': 'upper right', 'bbox_to_anchor': (0, 1)}
    colormaps_list = []

    if figsize is None:
        figsize = (3 + 0.15 * nb_probes, 3 + 0.2 * len(betas))

    if drop_na and min(betas.shape) > 1:
        handles, labels = [], []
        # convert categories to colors and get legends if specified
        if row_factors is not None:
            row_factors = [row_factors] if isinstance(row_factors, str) else row_factors
            if row_legends == '':
                row_legends = row_factors
            elif isinstance(row_legends, str):
                row_legends = [row_legends]
            subset = sheet[row_factors]
            row_factors, handles, labels, colormaps_list = _convert_df_values_to_colors(subset, row_legends)
        # plot the heatmap
        plot = sns.clustermap(betas, row_colors=row_factors, figsize=figsize, **heatmap_params)
        # reshape colorbar to be horizontal
        x0, _, _, _ = plot.cbar_pos
        plot.ax_cbar.set_position([x0, 0.9, plot.ax_row_dendrogram.get_position().width, 0.02])
        # add the legends if they exist
        leg_margin = 0
        if len(handles) > 0 and len(labels) > 0:
            plt.legend(handles=handles, labels=labels, **legend_params)
            leg_margin = -0.15

        if len(colormaps_list) > 0:
            fig = plot.fig       
            for i, (label, sm) in enumerate(colormaps_list):
                cbar_ax = fig.add_axes([-0.1+leg_margin, 0.9-(i+1)*0.23, 0.02, 0.2])  # [left, bottom, width, height]
                fig.colorbar(sm, cax=cbar_ax, label=label)

        # save plot
        if save_path is not None:
            plot.savefig(os.path.expanduser(save_path))
    else:
        if row_factors is not None:
            LOGGER.warning(f'Parameter {row_factors} is ignored when drop_na is False')

        plt.subplots(figsize=figsize)
        plot = sns.heatmap(betas, **heatmap_params)

        if save_path is not None:
            plot.get_figure().savefig(os.path.expanduser(save_path))


def _manhattan_plot(data_to_plot: pd.DataFrame, segments_to_plot: pd.DataFrame = None, chromosome_col='chromosome',
                    x_col='start', y_col='p_value', pval_col:str|None=None, log10=False, figsize:tuple[float, float]=(10,8),
                    annotation: Annotations | None = None, annotation_col: str = 'genes', nb_annotated_probes: int = 50,
                    sig_threshold: float | None = None, 
                    title: None | str = None, save_path: None | str=None) -> None:
    """Display a Manhattan plot of the given data.

    :param data_to_plot: dataframe to use for plotting. 
    :type data_to_plot: pandas.DataFrame

    :param segments_to_plot: if set, display the segments using columns "chromosome", "start", "end" and "mean_cnv" of
        the given dataframe, where start and end are the position on the chromosome (as returned by copy_number_segmentation())
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

    :param nb_annotated_probes: the number of probes to annotate (if annotation_col is provided). Defaul: 100
    :type nb_annotated_probes: int

    :param sig_threshold: threshold to display the significance line. If None, doesn't draw a line. Default: None
    :type sig_threshold: float |None

    :param log10: apply -log10 on the value column. Default: True
    :type log10: bool

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None"""

    if data_to_plot is None or len(data_to_plot) == 0:
        return

    # reset index as we might need to use the index as a column (e.g. to annotate probe ids)
    # data_to_plot = data_to_plot.reset_index()

    if x_col not in data_to_plot.columns or y_col not in data_to_plot.columns or chromosome_col not in data_to_plot.columns:
        LOGGER.error(f'Columns {x_col}, {y_col} and {chromosome_col} must be in the dataframe')
        return

    # remove non significant dots if specified
    if sig_threshold is not None:
        if pval_col is not None:
            data_to_plot = data_to_plot[data_to_plot[pval_col] < sig_threshold]
        else:
            LOGGER.warning('Cant find pvalue column to apply pvalue threshold on, drawing all DMRs')

    cols_to_keep = [x_col, y_col, chromosome_col]
    for col in ['probe_id', 'segment_id']:
        if col in data_to_plot.columns:
            cols_to_keep.append(col)

    data_to_plot = data_to_plot[cols_to_keep]
    data_to_plot = data_to_plot.dropna(subset=y_col)
    
    data_to_plot['merged_chr'] = merge_alt_chromosomes(data_to_plot[chromosome_col])
    # convert the chromosome column to int values
    if data_to_plot.dtypes[chromosome_col] is not int:
        data_to_plot['chr_id'] = get_chromosome_number(data_to_plot['merged_chr'], True)
        data_to_plot = data_to_plot.astype({'chr_id': 'int'})
    else:
        data_to_plot['chr_id'] = data_to_plot[chromosome_col]

    # figure initialization
    _, ax = plt.subplots(figsize=figsize)
    margin = int(max(data_to_plot[x_col]) / 10)
    chrom_start, chrom_end = 0, 0
    x_labels, x_major_ticks, x_minor_ticks = [], [], [0]

    # cant plot 0s
    zero_idxs = data_to_plot[y_col] == 0
    if sum(zero_idxs) > 0:
        new_min = np.min(data_to_plot.loc[~zero_idxs, y_col]) / 2
        data_to_plot.loc[zero_idxs, y_col] = new_min
        LOGGER.warning(f'{sum(zero_idxs)} probes = 0 found, replacing their value with {new_min}')

    # apply -log10 to p-values if needed
    if log10:
        data_to_plot[y_col] = -np.log10(data_to_plot[y_col])

    # define colormap and limits
    v_max = np.max(data_to_plot[y_col])
    v_min = np.min(data_to_plot[y_col])

    if v_min < 0:
        cmap = colormaps.get_cmap('Spectral')
        amplitude = min(abs(v_max), abs(v_min))  # center on 0
        v_max = amplitude
        v_min = - amplitude
    else:
        cmap = [colormaps.get_cmap('viridis_r'), colormaps.get_cmap('autumn_r')]
        v_min = 0

    # check annotation parameter, and select and clean up annotation if defined
    if annotation is not None:
        if annotation_col not in annotation.probe_infos.columns:
            LOGGER.error(f'{annotation_col} was not found in the annotation dataframe. '
                         f'Available columns : {annotation.probe_infos.columns}.')
        else:
            gene_info = annotation.probe_infos[['probe_id', annotation_col]].drop_duplicates().dropna()
            if data_to_plot.index.name == 'segment_id':
                data_to_plot = merge_dataframe_by(data_to_plot.merge(gene_info, on='probe_id').drop(columns='probe_id'),
                                                  ['start', 'chromosome'], observed=True)
                data_to_plot[annotation_col] = data_to_plot[annotation_col].apply(lambda x: ';'.join(set(x.split(';'))))
                data_to_plot = data_to_plot.reset_index()
            else:
                data_to_plot = data_to_plot.join(gene_info.set_index('probe_id'))
            
    data_to_plot = data_to_plot.reset_index(drop=True).drop_duplicates()
    # sort by chromosome and make the column a category
    data_to_plot = data_to_plot.sort_values(['chr_id', x_col]).astype({'chr_id': 'category'})

    # annotate the beta values with the biggest variation (negative or positive if beta)
    vals_to_sort = abs(data_to_plot[y_col]) if 'avg_beta' in y_col else data_to_plot[y_col]
    indexes_to_annotate = vals_to_sort.sort_values(ascending=False)[:nb_annotated_probes].index
    data_to_plot['annotate'] = False
    data_to_plot.loc[indexes_to_annotate, 'annotate'] = True
    
    # make indices for plotting
    data_to_plot_grouped = data_to_plot.groupby('chr_id', observed=True)

    # plot each chromosome scatter plot with its assigned color
    for idx_chr, (_, group) in enumerate(data_to_plot_grouped):
        # add margin to separate a bit the different groups; otherwise small groups won't show
        group[x_col] = chrom_start + group[x_col] + margin
        chrom_end = max(group[x_col]) + margin

        # build the chromosomes scatter plot
        curr_cmap = cmap[idx_chr % 2] if isinstance(cmap, list) else cmap
        ax.scatter(group[x_col], group[y_col], c=group[y_col], vmin=v_min, vmax=v_max, cmap=curr_cmap, alpha=1)
        
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

        # draw annotations for the first N dots, if annotation_col is set
        if annotation_col in group.columns:
            annot_col_idx = group.columns.get_loc(annotation_col)
            x_col_idx = group.columns.get_loc(x_col)
            y_col_idx = group.columns.get_loc(y_col)
            # shorten annotations that are too long
            long_anno_idxs = group[annotation_col].str.len() > 50
            group.loc[long_anno_idxs, annotation_col] = group.loc[long_anno_idxs, annotation_col].str.slice(0, 47) + '...'
            for row in group[group.annotate == True].itertuples(index=False):
                plt.annotate(' ' + row[annot_col_idx], (row[x_col_idx], row[y_col_idx]), c='black', annotation_clip=True)

        chrom_start = chrom_end

    ax.set_facecolor('#EBEBEB')  # set background color to grey
    [ax.spines[side].set_visible(False) for side in ax.spines]  # hide plot frame

    # grids style and plot limits
    ax.xaxis.grid(True, which='minor', color='white', linestyle='--')
    ax.yaxis.grid(True, color='white', alpha=0.9, linestyle='dotted')
    ax.set_axisbelow(True)  # so that the axis lines stay behind the dots
    ax.set_xlim((0 - margin, chrom_end + margin))

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
        what = 'bins' if segments_to_plot is not None else 'regions'
        title = f'Plotting {len(data_to_plot):,} {what}'
    plt.title(title)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))


def dmr_manhattan_plot(dm: DM, contrast: str | None = None,
                       chromosome_col='chromosome', x_col='start', y_col='p_value_adjusted',
                       annotation_col='genes', nb_annotated_probes: int=100, log10=True,
                       figsize:tuple[float, float]=(10, 8),
                       sig_threshold: float | None = 0.05, 
                       title: None | str = None, save_path: None | str=None) -> None:
    """Display a Manhattan plot of the given DM data

    :param dm: DM object with computed DMRs
    :type dm: pylluminator.DM

    :param contrast: name of the contrast to use.
    :type contrast: str

    :param chromosome_col: the name of the chromosome column in the `dmr` dataframe. Default: chromosome
    :type chromosome_col: str

    :param x_col: name of the column to use for X axis, start position of the probe/bin. Default: start
    :type x_col: str

    :param y_col: the name of the value column in the DMR dataframe. Default: p_value
    :type y_col: str

    :param annotation_col: the name of a column used to write annotation on the plots. Must be a column 
        in the Annotation data. The top 100 regions are annotated. Default: None
    :type annotation_col: str | None

    :param nb_annotated_probes: the number of probes to annotate (if annotation_col is provided). Defaul: 100
    :type nb_annotated_probes: int

    :param sig_threshold: Plot only DMRs below this significance threshold. Set to None to draw all DMRs. Default: 0.05
    :type sig_threshold: float | None

    :param log10: apply -log10 on the value column. Default: True
    :type log10: bool

    :param figsize: size of the figure. Default: (10, 8)
    :type figsize: tuple

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None
    :return: nothing"""

    if dm.dmr is None:
        LOGGER.error('No DMRs found')
        return None
    if dm.samples is None:
        LOGGER.error('No samples found in DM object')
        return None
    
    # get default contrast if none is specifier (works only if there is one contrast)
    if contrast is None:
        contrast = dm._get_default_contrast()
        if contrast is None:
            return

    pval_col = f'{contrast}_p_value_adjusted'

    data = dm.dmr.join(dm.segments.reset_index().set_index('segment_id'))

    if y_col not in data.columns:
        if f'{contrast}_{y_col}' not in data.columns:
            LOGGER.error(f'Column {y_col} not found in dmr dataframe. Existing columns : {data.columns}')
            return
        else:
            y_col = f'{contrast}_{y_col}'

    return _manhattan_plot(data_to_plot=data, chromosome_col=chromosome_col, y_col=y_col,
                           x_col=x_col, pval_col=pval_col, annotation=dm.samples.annotation, 
                           nb_annotated_probes=nb_annotated_probes,
                           annotation_col=annotation_col, sig_threshold=sig_threshold,
                           figsize=figsize, log10=log10, title=title, save_path=save_path)


def cns_manhattan_plot(data_to_plot: pd.DataFrame, segments_to_plot=None,
                       x_col='start_bin', chromosome_col='chromosome', y_col='cnv',
                       figsize:tuple[float, float]|None=(10, 8), title: None | str = None, save_path: None | str=None) -> None:
    """Display a Manhattan plot of the given CNS data, designed to work with the dataframes returned by
    copy_number_segmentation()

    :param data_to_plot: dataframe to use for plotting. Typically, the bins coordinate dataframe returned by
        copy_number_segmentation()
    :type data_to_plot: pandas.DataFrame

    :param segments_to_plot: if set, display the segments using columns "chromosome", "start", "end" and "mean_cnv" of
        the given dataframe, where start and end are the position on the chromosome (as returned by copy_number_segmentation())
    :type segments_to_plot: pandas.DataFrame

    :param chromosome_col: the name of the chromosome column in the `data_to_plot` dataframe. Default: chromosome
    :type chromosome_col: str

    :param x_col: name of the column to use for X axis, start position of the probe/bin. Default: start_bin
    :type x_col: str

    :param y_col: the name of the value column in the `data_to_plot` dataframe. Default: cnv
    :type y_col: str

    :param figsize: size of the figure. Default: (10, 8)
    :type figsize: tuple

    :param title: custom title for the plot. Default: None
    :type title: str | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None
    """

    _manhattan_plot(data_to_plot=data_to_plot, segments_to_plot=segments_to_plot, x_col=x_col,
                    chromosome_col=chromosome_col, y_col=y_col, title=title, figsize=figsize,
                    sig_threshold=None, log10=False, annotation=None, save_path=save_path)

########################################################################################################################

def show_chromosome_legend():
    """Display a legend for chromosome regions colors used in the 'visualize_gene' function, corresponding to Giemsa staining."""
    
    # chromosome region legend
    color_map = {'gene poor': 'lightgrey', 'moderate gene density': 'lightblue', 'intermediate gene density': 'blue', 'high gene density': 'darkblue', 'very high gene density': 'purple', 
                'variable gene density (often polymorphic)': 'lightgreen', 'centromere': 'yellow', 'stalk': 'pink'}
    #make a legend with the above colors
    patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    plt.figure(figsize=(8, 1))
    plt.legend(handles=patches, loc='center', ncol=len(patches))
    plt.axis('off')
    plt.title('Chromosome regions')
    plt.show()

def visualize_gene(samples: Samples, gene_name: str, apply_mask: bool=True, padding=1500, keep_na: bool=False,
                   protein_coding_only=True, custom_sheet: pd.DataFrame | None=None, var: None | str | list[str] = None,
                   figsize:tuple[float, float]=(15, 10), save_path: None | str=None,
                   row_factors: str | list[str] | None = None, row_legends: str | list[str] | None = '') -> None:
    """Show the beta values of a gene for all probes and samples in its transcription zone.

    :param samples: samples with beta values already calculated
    :type samples: Samples
    :param gene_name: name of the gene to visualize
    :type gene_name: str
    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool
    :param padding: length in kb pairs to add at the end and beginning of the transcription zone. Default: 1500
    :type: int
    :param keep_na: set to True to only output probes with no NA value for any sample. Default: False
    :type keep_na: bool
    :param protein_coding_only: limit displayed transcripts to protein coding ones. Default: True
    :type protein_coding_only: bool
    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display
    :type custom_sheet: pandas.DataFrame
    :param var: a column name or list of column names from the samplesheet to add to the heatmap labels. Default: None
    :type var: None | str | list[str]
    :param figsize: size of the whole plot. Default: (15, 15)
    :type figsize: tuple[float, float]
    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None
    :param row_factors: list of columns to show as color categories on the side of the heatmap. Must correspond to
        columns of the sample sheet. Default: None
    :type row_factors: str | list[str] | None
    :param row_legends: list of columns to generate a legend for. Set to None for no legends. Only work for columns also
        specified in row_factors. Default: '' (generate all legends)
    :type row_legends: str | list[str] | None

    :return: None
    """

    gene_data = samples.annotation.genome_info.transcripts_exons[samples.annotation.genome_info.transcripts_exons.gene_name == gene_name]

    # if the gene is not found in the transcripts file, get the position of the probes associated with the gene from the probe_info df
    if len(gene_data) == 0:
        LOGGER.warning(f'Gene {gene_name} not found in the transcript annotation data. Fetching probes in the region')
        regex = f'^(?:.*;)?(?:{gene_name})(?:;.*)?$'
        gene_data = samples.annotation.probe_infos[samples.annotation.probe_infos.genes.str.contains(regex) == True]  
        if len(gene_data) == 0:
            LOGGER.error(f'No probes associated with gene {gene_name} found')
            return
        start_pos = int(gene_data.start.min())
        end_pos = int(gene_data.end.max())
        LOGGER.info(f'Found {len(gene_data)} probes in the region, between pos {start_pos} and {end_pos}')
    else:
        start_pos = int(gene_data.transcript_start.min())
        end_pos = int(gene_data.transcript_end.max())

    start_pos -= padding
    end_pos += padding

    chromosome = gene_data.chromosome.drop_duplicates()

    if len(chromosome) > 1:
        LOGGER.error(f'This gene is associated with several chromosomes ({chromosome}).')
        return

    chromosome = str(chromosome.values[0])
    
    visualize_chromosome_region(samples, chromosome, start_pos, end_pos, apply_mask=apply_mask, keep_na=keep_na,
                                protein_coding_only=protein_coding_only, custom_sheet=custom_sheet, figsize=figsize,
                                save_path=save_path, var=var, gene_name=gene_name, padding=0, row_factors=row_factors,
                                row_legends=row_legends)

def _get_overlapping_rows(input_df: pd.DataFrame, start_pos:int, end_pos: int, start_col:str='start', end_col:str='end',
                          chr_col:str='chromosome', chromosome:str|int|None=None) -> pd.DataFrame | None:
    if start_col not in input_df.columns:
        LOGGER.error(f'Start column {start_col} not found in the dataframe')
        return None
    
    if end_col not in input_df.columns:
        LOGGER.error(f'End column {end_col} not found in the dataframe')
        return None
    
    if chromosome is not None and chr_col not in input_df.columns:
        LOGGER.error(f'Chromosome column {chr_col} not found in the dataframe')
        return None
    
    if chromosome is not None:
        if isinstance(chromosome, int):
            chromosome = str(chromosome)
        input_df = input_df[input_df[chr_col] == chromosome]
        if len(input_df) == 0:
            LOGGER.warning(f'Chromosome {chromosome} not found in the dataframe')
            return input_df
    
    return input_df[(input_df[start_col] <= end_pos) & (input_df[end_col] >= start_pos)]

def visualize_chromosome_region(samples: Samples, chromosome_id: int|str, start_pos: int, end_pos: int, apply_mask: bool=True, 
                                keep_na: bool=False, protein_coding_only=False, custom_sheet: pd.DataFrame | None=None, 
                                figsize:tuple[float, float]=(15, 10), save_path: None | str=None,
                                var: None | str | list[str] = None, gene_name: str|None = None, padding=0,
                                row_factors: str | list[str] | None = None, row_legends: str | list[str] | None = '') -> None:
    """Show the beta values of a gene for all probes and samples in its transcription zone.

    :param samples: samples with beta values already calculated
    :type samples: Samples
    
    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool
    
    :param keep_na: set to True to only output probes with no NA value for any sample. Default: False
    :type keep_na: bool
    :param protein_coding_only: limit displayed transcripts to protein coding ones. Default: False
    :type protein_coding_only: bool
    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display
    :type custom_sheet: pandas.DataFrame
    :param var: a column name or list of column names from the samplesheet to add to the heatmap labels. Default: None
    :type var: None | str | list[str]
    :param figsize: size of the whole plot. Default: (15, 15)
    :type figsize: tuple[float, float]
    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None
    :param row_factors: list of columns to show as color categories on the side of the heatmap. Must correspond to
        columns of the sample sheet. Default: None
    :type row_factors: str | list[str] | None
    :param row_legends: list of columns to generate a legend for. Set to None for no legends. Only work for columns also
        specified in row_factors. Default: '' (generate all legends)
    :type row_legends: str | list[str] | None
    :param padding: length in kb pairs to add at the end and beginning of the transcription zone. Default: 0
    :type: int
    :param gene_name: name of the gene, if applicable, to output in the title. Default: None
    :type gene_name: str | None
    :return: None
    """

    color_map = {'gneg': 'lightgrey', 'gpos25': 'lightblue', 'gpos50': 'blue', 'gpos75': 'darkblue', 'gpos100': 'purple',
                 'gvar': 'lightgreen', 'acen': 'yellow', 'stalk': 'pink'}

    links_args = {'color': 'red', 'alpha': 0.3}
    colormaps_list = []

    ################## DATA PREP

    # select chromosome information
    chromosome = str(chromosome_id).replace('chr', '')
    chr_df = samples.annotation.genome_info.chromosome_regions.loc[chromosome]

    # select transcripts information (transcripts that overlap with the selected region)
    gene_data = _get_overlapping_rows(samples.annotation.genome_info.transcripts_exons, start_pos, end_pos,
                                      start_col='transcript_start', end_col='transcript_end', chromosome=chromosome)    
    if gene_data is None or len(gene_data) == 0:
        LOGGER.warning(f'Found no transcript in region {start_pos} - {end_pos} for chromosome {chromosome}')
    else:
        if protein_coding_only:
            if 'transcript_type' not in gene_data.columns:
                LOGGER.info('Transcript type information not available, the protein_coding_only option is ignored')
            else:
                non_prot_coding = gene_data['transcript_type'] != 'protein_coding'
                if sum(non_prot_coding) > 0:
                    LOGGER.info(f'Dropping {sum(non_prot_coding)} non protein coding transcripts')
                    gene_data = gene_data[~non_prot_coding]
                if len(gene_data) == 0:
                    LOGGER.warning(f'No transcripts left in region {start_pos} - {end_pos} for chromosome {chromosome}')
                else:
                    gene_data = gene_data.sort_values('transcript_start')

    # select probes information (probes that overlap with the selected region)
    probes_data = _get_overlapping_rows(samples.annotation.probe_infos, start_pos, end_pos, chromosome=chromosome)
    if probes_data is None or len(probes_data) == 0:
        LOGGER.warning(f'No probes found in the interval {start_pos} - {end_pos} on chromosome {chromosome}')
    probes_data = probes_data[['probe_id', 'start', 'end']].drop_duplicates().set_index('probe_id')
    gene_betas = samples.get_betas(apply_mask=apply_mask, custom_sheet=custom_sheet)

    if gene_betas is None or len(gene_betas) == 0:
        LOGGER.error('No betas to plot')
        nb_probes = 0
        nb_samples = 0
        heatmap_data = None
    else:
        gene_betas = set_level_as_index(gene_betas, 'probe_id', drop_others=True)
        betas_location = gene_betas.join(probes_data, how='inner').sort_values('start')
        betas_data = betas_location if keep_na else betas_location.dropna()
        heatmap_data = betas_data.drop(columns=['start', 'end']).T
        nb_probes = len(betas_data)
        nb_samples = len(gene_betas.columns)

    ################## plot beta values heatmap (clustermap if no NA, heatmap otherwise)

    # chromosome, chr-transcript links, transcripts, transcript-betas links, betas heatmap
    nb_transcripts = len(set(gene_data.index))
    height_ratios = [0.02, 0.05, max(0.02, 0.02*nb_transcripts), 0.05, max(0.02, 0.02*nb_samples)]
    nb_plots = len(height_ratios)

    if heatmap_data is None or heatmap_data.empty:
        LOGGER.error('no beta data to plot')
        nb_plots -= 2
        height_ratios = height_ratios[:-2]
        _, axes = plt.subplots(figsize=figsize, nrows=nb_plots, height_ratios=height_ratios)
    else:
        label = samples.sample_label_name
        sheet = custom_sheet if custom_sheet is not None else samples.sample_sheet
        sheet = sheet.copy()[sheet[label].isin(heatmap_data.index)]  #  get intersection of the two dfs
        sheet = sheet.set_index(label)

        # add variable values to the column names
        if var is not None:
            if isinstance(var, str):
                var = [var]
            # update heatmap indexes names and sheet labels by adding values of 'var' columns after the sample names
            new_labels = [f'{c} ({",".join([str(sheet.loc[c, v]) for v in var])})' for c in heatmap_data.index]
            sheet = sheet.loc[heatmap_data.index,]  # sort sheet like headmap data
            sheet.index = new_labels  # update sheet index values
            sheet.index.name = label
            heatmap_data.index = new_labels

        heatmap_params = {'yticklabels': True, 'xticklabels': True, 'cmap': 'Spectral_r', 'vmin': 0, 'vmax': 1}
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=heatmap_params['cmap'], norm=norm)
        sm.set_array([])

        if keep_na:
            _, axes = plt.subplots(figsize=figsize, nrows=nb_plots, height_ratios=height_ratios)
            sns.heatmap(heatmap_data, ax=axes[-1], cbar=False, **heatmap_params)
            if row_factors is not None:
                LOGGER.warning('Parameter row_factors is ignored when keep_na is True')
                row_factors = None        
        else:
            handles, labels = [], []
            # convert categories to colors and get legends if specified
            if row_factors is not None:
                row_factors = [row_factors] if isinstance(row_factors, str) else row_factors
                if row_legends == '':
                    row_legends = row_factors
                elif isinstance(row_legends, str):
                    row_legends = [row_legends]
                subset = sheet[row_factors]
                row_factors, handles, labels, colormaps_list = _convert_df_values_to_colors(subset, row_legends) 
            dendrogram_ratio = 0.05
            g = sns.clustermap(heatmap_data, cbar_pos=None, figsize=figsize,  col_cluster=False, row_colors=row_factors,
                            dendrogram_ratio=(dendrogram_ratio, 0), **heatmap_params)
            # add the legends if they exist TODO recheck ca
            leg_margin = 0
            if len(handles) > 0 and len(labels) > 0:
                leg_margin = -0.15
                plt.legend(handles=handles, labels=labels, handler_map={str: _LegendTitle({'fontweight': 'bold'})},
                        loc='upper left', bbox_to_anchor=(-0.1 * len(row_factors.columns), 1))
            if len(colormaps_list) > 0:
                fig = g.fig       
                for i, (label, sm) in enumerate(colormaps_list):
                    cbar_ax = fig.add_axes([-0.1+leg_margin, 0.9-(i+1)*0.23, 0.02, 0.2])  # [left, bottom, width, height]
                    fig.colorbar(sm, cax=cbar_ax, label=label)
            shift_ratio = 1-(np.sum(height_ratios[:-1]) / np.sum(height_ratios))
            g.gs.update(top=shift_ratio, bottom=0)  # shift the heatmap to the bottom of the figure
            gs2 = gridspec.GridSpec(nb_plots - 1, 1, left=dendrogram_ratio + 0.005, bottom=shift_ratio,
                                    height_ratios=height_ratios[:-1])
            axes = [g.figure.add_subplot(gs) for gs in gs2]
            cbar_ax = axes[-1].inset_axes([-0.2, -0.7, 0.03, 0.6])

            # add a colorbar for the heatmap on the left of the heatmap
            cbar = g.figure.colorbar(sm, cax=cbar_ax)
            cbar.set_label('beta value', rotation=270, labelpad=15)

    ################## plot chromosome 

    chr_ax = axes[0]
    title = f'chromosome {chromosome}, pos {start_pos:,} - {end_pos:,}'
    if gene_name is not None:
        title = f'gene {gene_name} - ' + title
    chr_ax.set_title(title)

    # make rectangles of different colors depending on the chromosome region
    for _, row in chr_df.iterrows():
        curr_color = color_map[row.giemsa_staining]
        # if the region is a centromere, make it two triangles facing each other
        if row.giemsa_staining == 'acen':
            if 'name' in row and 'p' in row['name']:
                chr_ax.add_patch(mpatches.Polygon([[row.start, 0], [row.start + (row.end - row.start), 0.25], [row.start, 0.5]], color=curr_color))
            elif 'name' in row and 'q' in row['name']:
                chr_ax.add_patch(mpatches.Polygon([[row.start, 0.25], [row.end, 0], [row.end, 0.5]], color=curr_color))
            else:
                chr_ax.add_patch(mpatches.Rectangle((row.start, 0.05), row.end - row.start, 0.4, color=curr_color))
        else:
            chr_ax.add_patch(mpatches.Rectangle((row.start, 0), row.end - row.start, 0.5, color=curr_color))

    # red lines showing the beginning and end of the gene region
    chr_ax.plot([start_pos, start_pos], [-1, 1], **links_args)
    chr_ax.plot([end_pos, end_pos], [-1, 1], **links_args)

    chr_length = chr_df['end'].max()
    chr_ax.set_xlim(0, chr_length)
    chr_ax.set_ylim(0, 0.5)
    chr_ax.axis('off')

    ################## plot transcripts

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

        if 'transcript_strand' in transcript_data.columns:
            if transcript_data.iloc[0].transcript_strand == '-':
                arrow_coords = (transcript_start, y_position, -padding, 0)
                transcript_end += padding
            else:
                arrow_coords = (transcript_end, y_position, padding, 0)
                transcript_start -= padding
            # arrow at the end of the line to show the strand direction
            trans_ax.arrow(*arrow_coords, shape='full', fill=True, color='black', head_width=0.75, length_includes_head=True,
                        head_length=int(padding / 3), width=0, alpha=0.3)

        # line of the full transcript length
        trans_ax.plot([transcript_start, transcript_end], [y_position, y_position], color='black', alpha=0.3, zorder=1)

        # draw the patches for each transcript location
        for row in transcript_data.itertuples():
            trans_ax.add_patch(mpatches.Rectangle((row.transcript_start, y_position-0.5),
                                                  row.transcript_end - row.transcript_start, 1, color='black'))

            # # if a probe intersects with a transcript, draw a colored patch
            # for beta_row in betas_data.itertuples():
            #     if (row.start <= beta_row.start <= row.End) or (row.Start <= beta_row.end <= row.End):
            #         rec_coords = (beta_row.start, i*3+0.2), beta_row.end - beta_row.start
            #         trans_ax.add_patch(mpatches.Rectangle(*rec_coords, 0.6, color='limegreen', zorder=2))

    # transcript links
    chr_trans_link_ax = axes[1]

    if nb_probes > 0:
        for beta_row in betas_data.itertuples():
            rec_coords = (beta_row.start, 0), beta_row.end - beta_row.start, transcript_index * 2 + 1
            trans_ax.add_patch(mpatches.Rectangle(*rec_coords, zorder=2, **links_args))
            # draw link between chromosome and transcript
            trans_position = beta_row.start + (beta_row.end - beta_row.start) / 2
            trans_position_in_chr = trans_position / chr_length
            trans_position_in_trans = (trans_position - start_pos) / (end_pos - start_pos)
            chr_trans_link_ax.plot([trans_position_in_trans, trans_position_in_chr, trans_position_in_chr], [0, 0.8, 1], **links_args)
    else:
        # no beta data, draw the start and the end of the region
        position_in_chr = start_pos / chr_length
        position_in_trans = 0
        chr_trans_link_ax.plot([position_in_trans, position_in_chr, position_in_chr], [0, 0.8, 1], **links_args)
        position_in_chr = end_pos / chr_length
        position_in_trans = 1
        chr_trans_link_ax.plot([position_in_trans, position_in_chr, position_in_chr], [0, 0.8, 1], **links_args)

    # hide all axes but keep the y labels (transcript names)
    [trans_ax.spines[pos].set_visible(False) for pos in ['top', 'right', 'bottom', 'left']]
    trans_ax.set_yticks(y_positions, y_labels)
    trans_ax.set_xticks([])
    trans_ax.set_xlim(start_pos, end_pos)
    trans_ax.set_ylim(0, transcript_index * 2 + 1)

    chr_trans_link_ax.set_xlim(0, 1)
    chr_trans_link_ax.set_ylim(0, 1)
    chr_trans_link_ax.axis('off')

    ### plot transcript - beta values links

    if nb_probes > 0:
        lin_ax = axes[3]

        nb_factors = len(row_factors.columns) if row_factors is not None else 0

        probe_shift = nb_factors*0.03 + (1 - nb_factors*0.03) / (2 * nb_probes)

        for i, beta_row in enumerate(betas_data.itertuples()):
            probe_loc = beta_row.start - start_pos + (beta_row.end - beta_row.start) / 2
            x_transcript = probe_loc / (end_pos - start_pos)
            x_beta = (1 - nb_factors * 0.03) * i / nb_probes + probe_shift
            lin_ax.plot([x_beta, x_transcript, x_transcript], [0, 1.5, 2], **links_args)

        lin_ax.set_xlim(0, 1)
        lin_ax.set_ylim(0, 2)
        lin_ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))


def plot_betas_distribution(samples: Samples, group_column: str| None=None, what: list[str] | str = 'hyper',
                            annotation_column:str='cgi', orientation: str|None='h', custom_sheet:pd.DataFrame|None=None,
                            hypo_threshold:float=0.4, hyper_threshold:float=0.6,
                            save_path: None | str=None) -> None:
    """
    Plot the distribution of hyper (beta > 0.6) / hypo (beta < 0.4) methylated probes in the samples.

    :param samples: samples with beta values already calculated
    :type samples: Samples

    :param group_column: column name in the sample sheet to categorize the data vertically. Default: None
    :type group_column: str | None

    :param what: the metric to plot. Can be 'hypo', 'hyper', 'mid', 'nas' or 'all' for the 3 of them. Default: 'hyper'
    :type what: list[str] | str

    :param annotation_column: column name of the probe_infos dataframe to use to annotation probes (cgi, promoter_or_body..). Default: 'cgi'
    :type annotation_column: str

    :param orientation: 'h' or 'v', orientation of the plot. Default: 'h'
    :type orientation: str | None

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the samples to display
    :type custom_sheet: pandas.DataFrame | None

    :param hypo_threshold: max beta value to consider a probe hypo methylated. Default: 0.4
    :type hypo_threshold: float

    :param hyper_threshold: min beta value to consider a probe hyper methylated. Default: 0.6
    :type hyper_threshold: float

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None
    
    :return: None
    """
    # add CGI annotations to betas
    betas = samples.get_betas(custom_sheet=custom_sheet)
    if betas is None or len(betas) == 0:
        return None

    if annotation_column not in samples.annotation.probe_infos.columns:
        LOGGER.error('No CGI annotations found in the annotation data')
        return

    cgis = samples.annotation.probe_infos.set_index('probe_id')[annotation_column].dropna()
    cgi_betas = set_level_as_index(betas, 'probe_id', drop_others=True).join(cgis, how='inner')
    cgi_betas[annotation_column] = cgi_betas[annotation_column].apply(lambda x: x.split(';'))
    cgi_betas = cgi_betas.explode(annotation_column)

    # define aggregation functions
    def hypo(x):
        return 100 * sum(x < hypo_threshold) / len(x)

    def hyper(x):
        return 100 * sum(x > hyper_threshold) / len(x)

    def mid(x):
        return 100 * sum((hypo_threshold < x ) & (x < hyper_threshold)) / len(x)
    
    def nas(x):
        return 100 * np.count_nonzero(np.isnan(x)) / len(x)

    functions = {'hypo': hypo, 'hyper': hyper, 'mid': mid, 'nas': nas}
    if isinstance(what, str):
        what = functions.keys() if what == 'all' else [what]

    meth_prop = cgi_betas.groupby(annotation_column, observed=True).agg([functions[f] for f in what])
    meth_prop = pd.DataFrame(meth_prop.unstack()).reset_index()
    meth_prop.columns = [samples.sample_label_name, 'metric', annotation_column, 'proportion']

    if group_column is not None:
        if group_column not in samples.sample_sheet.columns:
            LOGGER.error(f'Column {group_column} not found in the sample sheet - ignoring parameter')
            return
        else:
            annot = samples.sample_sheet[[samples.sample_label_name, group_column]].drop_duplicates()
            meth_prop = meth_prop.merge(annot, on=samples.sample_label_name)

    hue = group_column if group_column is not None else 'metric'

    def rescale_axis(g: sns.FacetGrid, x_or_y = 'x'):
        # don't share the axis, but make them a bit more readable by having the same scale, and putting the dots 
        # on the left when mean % < 40 methylated probes, and on the right when mean % > 60
        axes = g.axes
        max_range =  0
        for ax_r in axes:
            for ax in ax_r:
                lims = ax.get_xlim() if x_or_y == 'x' else ax.get_ylim()
                max_range = max(max_range, lims[1] - lims[0])
        mid_range = 2 * max_range / 3
        
        for ax_r in axes:
            for ax in ax_r:
                lims = ax.get_xlim() if x_or_y == 'x' else ax.get_ylim()
                mean_axis = lims[1] - (lims[1] - lims[0])/2
                if mean_axis < 40:
                    new_lims = [lims[0],  lims[0] + 2 * mid_range]
                elif mean_axis > 60:
                    new_lims = [lims[1] - 2 * mid_range, lims[1]]
                else:
                    new_lims = [mean_axis - mid_range, mean_axis + mid_range]
                if x_or_y == 'x':
                    ax.set_xlim(new_lims)
                else:
                    ax.set_ylim(new_lims)

    if orientation == 'v':
        g = sns.catplot(data=meth_prop, x='proportion', y=group_column, row=annotation_column, col='metric', hue=hue,
                        kind='violin', fill=False, linewidth=1, inner='stick', density_norm='count',
                        sharex=False, height=2, aspect=2, orient='h', margin_titles=True, inner_kws={'alpha': 0.7})
        rescale_axis(g, 'x')
        g.set_axis_labels('', '')
        g.set_titles(row_template="{row_name}", col_template='Proportion of {col_name} methylated probes (%)')
    else:
        g = sns.catplot(data=meth_prop, y='proportion', x=group_column, col=annotation_column, row='metric', hue=hue,
                        kind='violin', fill=False, linewidth=1, inner='stick', density_norm='count',
                        sharey=False, height=4, aspect=1, orient='v', margin_titles=True, inner_kws={'alpha': 0.7})
        rescale_axis(g, 'y')
        g.set_axis_labels('', '')
        g.fig.suptitle(f'Proportion of {", ".join(what)} methylated probes (%)', y=1.05)
        g.set_titles(col_template="{col_name}", row_template='{row_name} methylated probes proportion (%)')

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

def plot_mean_beta_diff_per_group(samples: Samples, group_column: str, figsize=(5, 3),
                                  annotation_column:str='cgi', custom_sheet: pd.DataFrame|None = None,
                                  delta_beta_threshold:float=0.2, save_path: None | str=None) -> None:
    
    """Plot the distribution of hyper- and hypo- methylated probes in the samples. Compute the average beta values within each group of samples,
    and calculates the proportion of probes that have a significant methylation difference.

    :param samples: samples with beta values already calculated. NA beta values are dropped (i.e. masks are applied)
    :type samples: Samples

    :param group_column: column name of the sample metadata from the sample sheet used to categorize the data, e.g. Phenotype.
      It only works with metadata that has 2 possible values (e.g. Control vs Patients)
    :type group_column: str

    :param annotation_column: column name of the probe_infos dataframe to use to annotation probes (cgi, promoter_or_body..). Default: 'cgi'
    :type annotation_column: str

    :param delta_beta_threshold: minimum difference of average beta value between the two groups to consider that a probe is hyper- or -hypo methylated. A float between 0 and 1. Default: 0.2
    :type delta_beta_threshold: float

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: None
    """
    # check input parameters

    betas = samples.get_betas(custom_sheet=custom_sheet).dropna()
    if betas is None or len(betas) == 0:
        return None

    if annotation_column not in samples.annotation.probe_infos.columns:
        LOGGER.error(f'Column {annotation_column} not found in the annotation data. Available columns: {samples.annotation.probe_infos.columns}')
        return

    if group_column not in samples.sample_sheet.columns:
        LOGGER.error(f'Column {group_column} not found in the sample sheet')
        return
    
    if not 0 < delta_beta_threshold < 1:
        LOGGER.error('delta_beta_threshold must be betweend 0 and 1')
        return
    
    # get samples metadata to group samples
    sample_metadata = samples.sample_sheet[[samples.sample_label_name, group_column]].drop_duplicates().set_index(samples.sample_label_name)
    sample_metadata_values = sample_metadata[group_column].unique()
    if len(sample_metadata_values) != 2:
        LOGGER.error(f'you need exactly two groups (e.g. control vs patients). Detected groups: {sample_metadata_values}')
        return
    
    # replace samples labels by their respective group
    betas.columns = [sample_metadata.loc[c, group_column] for c in betas.columns]
    betas.columns.name = group_column
    
    # compute the average beta value per group, and the difference between each group
    avg_beta_per_group = betas.T.groupby(by=group_column).mean().T
    diff_beta_btw_group = avg_beta_per_group[sample_metadata_values[0]] - avg_beta_per_group[sample_metadata_values[1]]

    # add CGI annotations to betas
    cgis = samples.annotation.probe_infos.set_index('probe_id')[annotation_column].dropna()
    cgi_betas = set_level_as_index(diff_beta_btw_group, 'probe_id', drop_others=True).join(cgis, how='inner')
    if cgi_betas[annotation_column].dtype == 'object':
        cgi_betas[annotation_column] = cgi_betas[annotation_column].apply(lambda x: x.split(';'))
    cgi_betas = cgi_betas.explode(annotation_column)

    # define aggregation functions
    def hypo(x):
        return 100 * sum(x < - delta_beta_threshold) / len(x)

    def hyper(x):
        return 100 * sum(x > delta_beta_threshold) / len(x)
    
    meth_prop = cgi_betas.groupby(annotation_column, observed=True).agg([hypo, hyper])
    meth_prop = meth_prop.droplevel(0, axis=1)
    meth_prop.columns = [f'hypomethylated probes ({sample_metadata_values[0]} < {sample_metadata_values[1]})',
                         f'hypermethylated probes ({sample_metadata_values[0]} > {sample_metadata_values[1]})',]
    
    y_lim = [0, meth_prop.max(axis=None)*1.2]
    g = meth_prop.plot.bar(ylabel='% probes', ylim=y_lim, figsize=figsize, color=[ '#F8766D', '#1f77b4'] )
    g.set_title(f'Proportion of differentially methylated probes (mean delta beta threshold : {delta_beta_threshold})')
    sns.move_legend(g.axes, 'upper left', bbox_to_anchor=(1, 1))

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

def analyze_replicates(samples: Samples, sample_id_column: str, replicate_names: list[str] = None,  return_df=False,
                       xlim: None|tuple[float, float]=None, save_path: str =None, figsize=(10, 5), **kwargs) -> pd.DataFrame | None:
    """ Analyze the beta values standard deviation of the technical replicates to check for batch effect or quality
    issues.

    :param samples: samples with beta values already calculated
    :type samples: Samples


    :param sample_id_column: the name of the column in the sample sheet that contains the sample ids, used to identify the replicates
    :type sample_id_column: str

    :param replicate_names: list of the replicate names to analyze. If None, analyze all samples. Default: None
    :type replicate_names: list[str] | None

    :param return_df: if True, return the dataframe used to plot the graph. Default: False
    :type return_df: bool

    :param xlim: set the x-axis limits. Default: None
    :type xlim: tuple[float, float] | None

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :return: the dataframe with beta values standard deviation per replicate, or None if return_df is False
    :rtype: pandas.DataFrame | None
    """

    if replicate_names is None:
        replicate_names = samples.sample_labels

    # select the specified samples
    samples = samples.copy()
    samples.reset_poobah()  # remove p_values columns
    samples.subset(replicate_names)
    sheet = samples.sample_sheet

    # get the beta values standard deviation for each probe
    beta_df = samples.get_betas()
    beta_std_df_list = {}
    for name, group in sheet.groupby(sample_id_column):
        replicate_names = group[samples.sample_label_name].values.tolist()
        beta_std_df_list[name] = beta_df[replicate_names].std(axis=1)
    beta_std_df = pd.concat(beta_std_df_list, axis=1)

    # df should be in long format
    replicate_ids = beta_std_df.columns
    beta_std_df = beta_std_df.reset_index().drop(columns=['probe_id', 'probe_type'])
    long_df = pd.melt(beta_std_df, id_vars=['type', 'channel'], value_vars=replicate_ids, var_name='Replicate',  value_name='Value')
    long_df['channel'] = long_df['channel'].cat.add_categories('').fillna('').astype(str)
    long_df['channel'] = 'type ' + long_df['type'].astype(str) + ' ' + long_df['channel']

    # plot
    plt.style.use('ggplot')
    g = sns.catplot(long_df, kind='violin', hue='Replicate', y='channel', x='Value', height=figsize[1], aspect=figsize[0]/figsize[1], **kwargs)
    g.set_axis_labels('beta values standard deviation', 'probes channel')

    if xlim is not None:
        g.set(xlim=xlim)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

    if return_df:
        return long_df.groupby(['Replicate', 'channel']).describe()

    return None

def _select_metadata(input_data: Samples | pd.DataFrame, columns:list[str]|None=None,
                     numeric_only=False, keep_col: str|None=None) -> pd.DataFrame | None:
    if isinstance(input_data, Samples):
        metadata = input_data.sample_sheet.drop(columns=input_data.sample_label_name)
    elif isinstance(input_data, pd.DataFrame):
        metadata = input_data.copy()
    else:
        LOGGER.error('input_data must be a Samples object or a pandas DataFrame')
        return None

    if keep_col is not None and keep_col not in metadata.columns:
        LOGGER.error(f'Column {keep_col} not found in the metadata')
        return None

    if numeric_only:
        valid_columns = metadata.select_dtypes(include=[np.number]).columns

        if len(valid_columns) == 0:
            LOGGER.error('No numeric columns to plot')
            return None
    else:
        valid_columns = metadata.columns

    if columns is not None:
        selected_cols = []
        for c in columns:
            if c not in valid_columns:
                LOGGER.warning(f'Column {c} not found in the metadata, ignoring it')
            else:
                selected_cols.append(c)
        if len(selected_cols) == 0:
            LOGGER.error('No valid columns to plot')
            return None

        if keep_col is not None:
            selected_cols.append(keep_col)
        metadata = metadata[selected_cols]

    return metadata

def metadata_correlation(input_data: Samples | pd.DataFrame, columns:list[str]|None=None, abs_corr=True, save_path=None) -> None:
    """Plot the correlation between the metadata columns of the samplesheet.

    :param input_data: Samples object to analyze the sample sheet of, or a sample sheet dataframe directly
    :type input_data: Samples | pd.DataFrame

    :param columns: list of columns to use for the correlation. If None, use all columns. Default: None
    :type columns: list[str] | None

    :param abs_corr: if True, plot the absolute correlation. Default: True
    :type abs_corr: bool

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None
    """
    metadata = _select_metadata(input_data, columns)
    if metadata is None:
        return

    plt.style.use('ggplot')

    corr = metadata.apply(lambda x:  x if is_numeric_dtype(x) else pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

    fisize = max(5.0, len(metadata.columns)/1.5)
    plt.subplots(figsize=(fisize,fisize - 1))
    cmap = sns.cm.rocket_r if abs_corr else 'vlag'
    vmin = 0 if abs_corr else -1
    sns.heatmap(corr, annot=True, fmt=".2f", vmax=1, vmin=vmin, cmap=cmap)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))

def metadata_pairplot(input_data: Samples | pd.DataFrame, columns:list[str]|None=None,
                      hue: str | None = None, kind='reg', corner=True, height=1.5,
                      save_path=None, **kwargs) -> None:
    """ Build a pair plot from the samples' sample sheet. Just a very simple wrapper around sns.pairplot()

    :param input_data: Samples object to analyze the sample sheet of, or a sample sheet dataframe directly
    :type input_data: Samples | pd.DataFrame

    :param columns: list of columns to use for the correlation. If None, use all columns. Default: None
    :type columns: list[str] | None

    :param hue: column name to use for the color of the points and categorize them for regression. Default: None
    :type hue: str | None

    :param kind: kind of plot to use. Possible values: 'scatter', 'kde', 'hist', 'reg'. Default: 'reg'
    :type kind: str

    :param corner: if True, plot only the lower triangle of the matrix. Default: True
    :type corner: bool

    :param height: height of each facet in inches. Default: 1.5
    :type height: float

    :param save_path: if set, save the graph to save_path. Default: None
    :type save_path: str | None

    :param kwargs: additional arguments to pass to sns.pairplot()
    :return: None
    """

    metadata = _select_metadata(input_data, columns, True, keep_col=hue)
    if metadata is None:
        return

    plt.style.use('ggplot')

    sns.pairplot(metadata, height=height, corner=corner, kind=kind, hue=hue, **kwargs)

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path))