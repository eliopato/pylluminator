import numpy as np

from sklearn.manifold import MDS
from sklearn.decomposition import PCA, DictionaryLearning, FactorAnalysis, FastICA, IncrementalPCA, KernelPCA
from sklearn.decomposition import LatentDirichletAllocation, MiniBatchDictionaryLearning, MiniBatchNMF
from sklearn.decomposition  import NMF, SparsePCA, TruncatedSVD, MiniBatchSparsePCA

from pylluminator.samples import Samples

from pylluminator.utils import get_logger

LOGGER = get_logger()

def dimensionality_reduction(samples: Samples,  model='PCA', nb_probes: int | None=1000, apply_mask=False, custom_sheet=None, **kwargs):
    """Plot samples in 2D space according to their beta distances.

    :param samples : samples to plot
    :type samples: Samples

    :param model: identifier of the model to use. Available models are 'PCA': PCA, 'MDS': MDS, 'DL': DictionaryLearning,
        'FA': FactorAnalysis, 'FICA': FastICA, 'IPCA': IncrementalPCA, 'KPCA': KernelPCA, 'LDA': LatentDirichletAllocation,
        'MBDL': MiniBatchDictionaryLearning, 'MBNMF': MiniBatchNMF, 'MBSPCA': MiniBatchSparsePCA, 'NMF': NMF,
        'SPCA': SparsePCA, 'TSVD': TruncatedSVD. Default: 'PCA'
    :type model: str

    :param nb_probes: number of probes to use for the model, selected from the probes with the most beta variance.
        If None, use all the probes. Default: 1000
    :type nb_probes: int | None

    :param apply_mask: True removes masked probes from betas, False keeps them. Default: True
    :type apply_mask: bool

    :param custom_sheet: a sample sheet to use. By default, use the samples' sheet. Useful if you want to filter the
        samples to display. Default: None
    :type custom_sheet: pandas.DataFrame | None

    :param kwargs: parameters passed to the model

    :return: scikit learn model, the fitted model, the samples' names, the number of probes used"""

    models = {'PCA': PCA, 'MDS': MDS, 'DL': DictionaryLearning, 'FA': FactorAnalysis, 'FICA': FastICA,
              'IPCA': IncrementalPCA, 'KPCA': KernelPCA, 'LDA': LatentDirichletAllocation,
              'MBDL': MiniBatchDictionaryLearning, 'MBNMF': MiniBatchNMF, 'MBSPCA': MiniBatchSparsePCA, 'NMF': NMF,
               'SPCA': SparsePCA, 'TSVD': TruncatedSVD}

    if model not in models.keys():
        LOGGER.error(f'Unknown model {model}. Known models are {models.keys()}')
        return None, None, None, None

    sk_model = models[model]

    # get betas with or without masked probes and samples
    betas = samples.get_betas(apply_mask=apply_mask, custom_sheet=custom_sheet)

    if betas is None or len(betas) == 0:
        LOGGER.error('No betas to plot')
        return None, None, None, None

    if model in ['PCA'] and 'n_components' in kwargs and kwargs['n_components'] > len(betas.columns):
        LOGGER.error(f'Number of components {kwargs["n_components"]} is greater than the number of samples {len(betas.columns)}')
        return None, None, None, None

    # get betas with the most variance across samples
    betas_variance = np.var(betas.dropna() , axis=1)  # remove NA
    nb_probes = len(betas_variance) if nb_probes is None else min(nb_probes, len(betas_variance))
    indexes_most_variance = betas_variance.sort_values(ascending=False)[:nb_probes].index
    betas_most_variance = betas.loc[indexes_most_variance]

    # fit the model
    model_ini = sk_model(**kwargs)
    fit = model_ini.fit_transform(betas_most_variance.T)

    return model_ini, fit, betas.columns, nb_probes