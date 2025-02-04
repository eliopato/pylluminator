"""Classes that handles probes masks"""
import pandas as pd

from pylluminator.utils import get_logger

LOGGER = get_logger()

class Mask:
    """
    A mask is a set of probes that are masked for a specific sample or for all samples.

    :var mask_name: the name of the mask
    :vartype mask_name: str
    :var sample_label: the name of the sample the mask is applied to
    :vartype sample_label: str
    :var series: a pandas Series of booleans, where True indicates that the probe is masked
    :vartype series: pandas.Series
    """
    def __init__(self, mask_name: str, sample_label: str | None, series: pd.Series):
        """Create a new Mask object.
        
        :param mask_name: the name of the mask
        :type mask_name: str
        :param sample_label: the name of the sample the mask is applied to. Default: None
        :type sample_label: str | None
        :param series: a pandas Series of booleans, where True indicates that the probe is masked
        :type series: pandas.Series"""
        self.mask_name = mask_name
        self.sample_label = sample_label
        if not isinstance(series, pd.Series):
            raise ValueError("series must be a pandas Series.")
        self.series = series

    def __str__(self):
        scope_str = f'sample {self.sample_label}' if self.sample_label is not None else 'all samples'
        return f"Mask(name: {self.mask_name}, scope: {scope_str}, # masked probes: {sum(self.series):,})"

    def __repr__(self):
        return self.__str__()

    # define a copy method
    def copy(self):
        """Creates a copy of the Mask object."""
        return Mask(self.mask_name, self.sample_label, self.series.copy())

class MaskCollection:
    """A collection of masks, each mask is a set of probes that are masked for a specific sample or for all samples.

    :var masks: a dictionary of masks, where the key is a tuple (mask_name, sample_label) and the value is a Mask object
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

        if (mask.mask_name, mask.sample_label) in self.masks:
            LOGGER.info(f"{mask} already exists, overriding it.")

        self.masks[(mask.mask_name, mask.sample_label)] = mask

    def get_mask(self, mask_name: str | list[str] | None=None, sample_label: str | list[str] | None=None) -> pd.Series | None:
        """Retrieve a mask by name and scope. If no sample_label is defined, return the mask that applies to all
        samples, without considering masks of specific samples. If a sample_label is defined, the mask includes the
        masks that applies to all samples combined with the masks of the samples(s) defined. If one or more mask_name
        are defined, only these masks will be considered.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | list[str] | None
        :param sample_label: the name of the sample the mask is applied to. Default: None
        :type sample_label: str | None

        :return: a pandas Series of booleans, where True indicates that the probe is masked
        :rtype: pandas.Series | None"""

        # to get a specific sample mask, retrieve the common mask and apply the sample mask on top of it
        mask_series = None

        if isinstance(sample_label, list):
            for si in sample_label:
                si_mask = self.get_mask(mask_name, si)
                mask_series = si_mask if mask_series is None else mask_series | si_mask

        elif isinstance(sample_label, str):
            mask_series = self.get_mask(mask_name, None)

        if isinstance(mask_name, str):
            mask_name = [mask_name]

        for mask in self.masks.values():
            if mask_name is None or mask.mask_name in mask_name:
                if mask.sample_label == sample_label:
                    mask_series = mask.series if mask_series is None else mask_series | mask.series

        return mask_series

    def get_mask_names(self, sample_label: str | list[str] | None) -> set:
        """Return the names of the masks existing for specific sample(s)

        :return: the mask names
        :rtype: set"""
        names = set()
        if isinstance(sample_label, str):
            sample_label = [sample_label]
        for mask in self.masks.values():
            if mask.sample_label in sample_label:
                names.add(mask.mask_name)
        return names

    def number_probes_masked(self, mask_name: str | None =None, sample_label: str| None=None) -> int:
        """Return the number or masked probes for a specific sample or for all samples if no sample name is provided.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | None
        :param sample_label: the name of the sample the mask is applied to. Default: None
        :type sample_label: str | None

        :return: number of masked probes
        :rtype: int"""

        mask = self.get_mask(mask_name, sample_label)
        if mask is None:
            return 0
        return sum(mask)

    def reset_masks(self):
        """Reset all masks."""
        self.masks = {}

    def remove_masks(self, mask_name : str | list[str] | None=None, sample_label: str | list[str] | None=None) -> None:
        """Reset the mask for specific samples or for all samples if no sample name is provided. If a mask name is
        provided,only delete this mask.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | list[str] | None
        :param sample_label: the name(s) of the sample(s) the mask is applied to. Default: None
        :type sample_label: str | list[str] | None

        :return: None
        """
        if isinstance(mask_name, str):
            mask_name = [mask_name]
        if isinstance(sample_label, str):
            sample_label = [sample_label]

        if sample_label is None and mask_name is None:
            self.reset_masks()
        elif sample_label is None:
            # remove all masks with the given name(s)
            self.masks = {k: v for k, v in self.masks.items() if v.mask_name not in mask_name}
        elif mask_name is None:
            # remove all masks for the given sample(s)
            self.masks = {k: v for k, v in self.masks.items() if v.sample_label not in sample_label}
        else:
            # remove a specific mask
            for si in sample_label:
                for mn in mask_name:
                    self.masks.pop((mn, si), None)

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