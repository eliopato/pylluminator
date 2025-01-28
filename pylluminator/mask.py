"""Classes that handles probes masks"""
import pandas as pd

from pylluminator.utils import get_logger

LOGGER = get_logger()

class Mask:
    """
    A mask is a set of probes that are masked for a specific sample or for all samples.

    :var mask_name: the name of the mask
    :vartype mask_name: str
    :var sample_name: the name of the sample the mask is applied to
    :vartype sample_name: str
    :var series: a pandas Series of booleans, where True indicates that the probe is masked
    :vartype series: pd.Series
    """
    def __init__(self, mask_name: str, sample_name: str | None, series: pd.Series):
        """Create a new Mask object.
        
        :param mask_name: the name of the mask
        :type mask_name: str
        :param sample_name: the name of the sample the mask is applied to. Default: None
        :type sample_name: str | None
        :param series: a pandas Series of booleans, where True indicates that the probe is masked
        :type series: pd.Series"""
        self.mask_name = mask_name
        self.sample_name = sample_name
        if not isinstance(series, pd.Series):
            raise ValueError("series must be a pandas Series.")
        self.series = series

    def __str__(self):
        scope_str = f'sample {self.sample_name}' if self.sample_name is not None else 'all samples'
        return f"Mask(name: {self.mask_name}, scope: {scope_str}, # masked probes: {sum(self.series):,})"

    def __repr__(self):
        return self.__str__()

    # define a copy method
    def copy(self):
        """Creates a copy of the Mask object."""
        return Mask(self.mask_name, self.sample_name, self.series.copy())

class MaskCollection:
    """A collection of masks, each mask is a set of probes that are masked for a specific sample or for all samples.

    :var masks: a dictionary of masks, where the key is a tuple (mask_name, sample_name) and the value is a Mask object
    :vartype masks: dict
    """
    def __init__(self):
        self.masks = {}

    def add_mask(self, mask: Mask) -> None:
        """Add a new mask to the collection.

        :param mask: the mask to add
        :type mask: Mask"""
        if not isinstance(mask, Mask):
            raise ValueError("mask must be an instance of Mask.")

        if mask.series is None:
            LOGGER.info(f"{mask} has no masked probes.")
            return None

        if (mask.mask_name, mask.sample_name) in self.masks:
            LOGGER.info(f"{mask} already exists, overriding it.")

        self.masks[(mask.mask_name, mask.sample_name)] = mask

    def get_mask(self, mask_name: str | None =None, sample_name: str| None=None) -> pd.Series | None:
        """Retrieve a mask by name and scope.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | None
        :param sample_name: the name of the sample the mask is applied to. Default: None
        :type sample_name: str | None

        :return: a pandas Series of booleans, where True indicates that the probe is masked
        :rtype: pd.Series | None"""

        # to get a specific sample mask, retrieve the common mask and apply the sample mask on top of it
        if sample_name is None:
            mask_series = None
        else:
            mask_series = self.get_mask(mask_name, None)

        for mask in self.masks.values():
            if mask_name is None or mask.mask_name == mask_name:
                if mask.sample_name == sample_name:
                    if mask_series is None:
                        mask_series = mask.series
                    else:
                        mask_series = mask_series | mask.series

        return mask_series

    def number_probes_masked(self, mask_name: str | None =None, sample_name: str| None=None) -> int:
        """Return the number or masked probes for a specific sample or for all samples if no sample name is provided.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | None
        :param sample_name: the name of the sample the mask is applied to. Default: None
        :type sample_name: str | None

        :return: number of masked probes
        :rtype: int"""

        mask = self.get_mask(mask_name, sample_name)
        if mask is None:
            return 0
        return sum(mask)

    def reset_masks(self):
        """Reset all masks."""
        self.masks = {}

    def remove_masks(self, mask_name : str | None=None, sample_name: str| None=None) -> None:
        """Reset the mask for a specific sample or for all samples if no sample name is provided.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | None
        :param sample_name: the name of the sample the mask is applied to. Default: None
        :type sample_name: str | None

        :return: None
        """
        if sample_name is None and mask_name is None:
            self.reset_masks()
        elif sample_name is None:
            self.masks = {k: v for k, v in self.masks.items() if v.mask_name != mask_name}  # remove all masks with the given name
        elif mask_name is None:
            self.masks = {k: v for k, v in self.masks.items() if v.sample_name != sample_name}  # remove all masks for the given sample
        else:
            self.masks.pop((mask_name, sample_name), None)  # remove a specific mask

    def copy(self):
        """Creates a copy of the MaskCollection object."""
        new_mask_collection = MaskCollection()
        for mask in self.masks.values():
            new_mask_collection.add_mask(mask.copy())

        return new_mask_collection

    def __str__(self):
        desc = ''
        for mask in self.masks.values():
            desc += mask.__str__() + '\n'
        return desc

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item: int | str) -> Mask | None:
        if isinstance(item, str):
            return self.get_mask(mask_name=item)

        if isinstance(item, int) and item < len(self.masks):
            return list(self.masks.values())[item]

        return None