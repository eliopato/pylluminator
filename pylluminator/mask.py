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
    :var indexes: a MultiIndex of masked probes
    :vartype indexes: pandas.MultiIndex
    """
    def __init__(self, mask_name: str, sample_label: str | None, indexes: pd.MultiIndex):
        """Create a new Mask object.
        
        :param mask_name: the name of the mask
        :type mask_name: str
        :param sample_label: the name of the sample the mask is applied to. Default: None
        :type sample_label: str | None
        :param indexes: a pandas MultiIndex of probes to mask
        :type indexes: pandas.MultiIndex"""
        self.mask_name = mask_name
        self.sample_label = sample_label
        if not isinstance(indexes, pd.MultiIndex):
            raise ValueError("indexes must be a pandas MultiIndex.")
        self.indexes = indexes

    def __str__(self):
        scope_str = f'sample {self.sample_label}' if self.sample_label is not None else 'all samples'
        return f"Mask(name: {self.mask_name}, scope: {scope_str}, # masked probes: {len(self.indexes):,})"

    def __repr__(self):
        return self.__str__()

    # define a copy method
    def copy(self):
        """Creates a copy of the Mask object."""
        return Mask(self.mask_name, self.sample_label, self.indexes.copy())

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

        if mask.indexes is None or len(mask.indexes) == 0:
            LOGGER.info(f"{mask} has no masked probes, skipping it.")
            return None

        if (mask.mask_name, mask.sample_label) in self.masks:
            LOGGER.info(f"{mask} already exists, overriding it.")

        self.masks[(mask.mask_name, mask.sample_label)] = mask

    def get_mask(self, mask_name: str | list[str] | None=None, sample_label: str | list[str] | None=None) -> pd.MultiIndex | None:
        """Retrieve a mask by name and scope. If no sample_label is defined, return the mask that applies to all
        samples, without considering masks of specific samples. If one or more sample_labels are defined, the mask
        includes the masks that applies to all samples combined with the masks of the samples(s) defined. If one or more
        mask_name are defined, only these masks will be considered.

        :param mask_name: the name of the mask. Default: None
        :type mask_name: str | list[str] | None
        :param sample_label: the name of the sample the mask is applied to. Default: None
        :type sample_label: str | None

        :return: a pandas Series of booleans, where True indicates that the probe is masked
        :rtype: pandas.Series | None"""

        mask_indexes_dfs = []

        if isinstance(mask_name, str):
            mask_name = [mask_name]

        if isinstance(sample_label, str):
            sample_label = [sample_label]

        for mask in self.masks.values():
            if mask_name is None or mask.mask_name in mask_name:
                if mask.sample_label is None or (sample_label is not None and mask.sample_label in sample_label):
                    mask_indexes_dfs.append(mask.indexes.to_frame().reset_index(drop=True))

        if len(mask_indexes_dfs) == 0:
            return None

        return pd.concat(mask_indexes_dfs).drop_duplicates().set_index(['type', 'channel', 'probe_type', 'probe_id']).index

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
        return len(mask)

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

    def __iter__(self):
        return iter(self.masks.values())