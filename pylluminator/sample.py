# """
# Class that holds the methylation information of a sample parsed from an idat file, as well as mask information
# and annotation metadata
# """
#
# import gc
# import re
# import os
# from pathlib import Path
#
# import pandas as pd
# import numpy as np
# from statsmodels.distributions.empirical_distribution import ECDF as ecdf
#
# from pylluminator.read_idat import IdatDataset
# from pylluminator.annotations import Annotations, Channel, ArrayType, detect_array, GenomeVersion
# from pylluminator.stats import norm_exp_convolution, quantile_normalization_using_target, background_correction_noob_fit
# from pylluminator.stats import iqr
# from pylluminator.utils import get_column_as_flat_array, mask_dataframe, save_object, load_object, remove_probe_suffix
#
# from pylluminator.utils import get_logger
#
# LOGGER = get_logger()
#
#
# class Sample:
#     """This class holds all methylation information of a sample in a `signal_df` data frame, as well as the current
#     mask information in `masked_indexes`, a multi-index of all the currently masked probes.
#
#     :ivar idata: raw data as read from .idat files. Dictionary with Channel as keys
#     :vartype idata: dict
#
#     :ivar name: sample name used as identifier.
#     :vartype name: str
#
#     :ivar annotation: probes metadata. Default: None
#     :vartype annotation: Annotations | None
#
#     :ivar masked_indexes: list of indexes masked for this sample
#     :vartype masked_indexes: pandas.MultiIndex
#
#     :ivar min_beads: required minimum number of beads per probe
#     :vartype min_beads: int | None
#     """
#
#     ####################################################################################################################
#     # Initialization
#     ####################################################################################################################
#
#     def __init__(self, name: str):
#         """Initialize the object, with only its name - every other attributes are set to None
#
#         :param name: sample name
#         :type name: str"""
#         self.idata = None
#         self._signal_df = None
#         self._betas_df = None
#         self.name = name
#         self.masked_indexes = None
