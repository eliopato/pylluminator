import pandas as pd
from pylluminator.utils import get_logger

LOGGER = get_logger()

class Mask:
    def __init__(self, mask_name: str, sample_name: str | None, series: pd.Series):
        self.mask_name = mask_name
        self.sample_name = sample_name
        self.series = series

    def __str__(self):
        scope_str = f'sample {self.sample_name}' if self.sample_name is not None else 'all samples'
        return f"Mask(name: {self.mask_name}, scope: {scope_str}, # masked probes: {sum(self.series):,})"

    def __repr__(self):
        return self.__str__()

    # define a copy method
    def copy(self):
        return Mask(self.mask_name, self.sample_name, self.series.copy())

class MaskCollection:
    def __init__(self):
        self.masks = {}

    def add_mask(self, mask: Mask) -> None:
        """Add a new mask to the collection."""
        if not isinstance(mask, Mask):
            raise ValueError("mask must be an instance of Mask.")

        if (mask.mask_name, mask.sample_name) in self.masks:
            LOGGER.warning(f"{mask} already exists, overriding it.")

        if mask.series is None:
            LOGGER.warning(f"{mask} has no masked probes.")
            return None

        self.masks[(mask.mask_name, mask.sample_name)] = mask

    def get_mask(self, mask_name=None, sample_name: str| None=None) -> pd.Series | None:
        """Retrieve a mask by name and scope."""
        # to get a specific sample mask, retrieve the common mask and apply the sample mask on top of it
        if sample_name is None:
            mask_series = None
        else:
            mask_series = self.get_mask(mask_name, None)

        for mask in self.masks.values():
            if mask_name is None or mask.mask_name == mask_name:
                if sample_name is None or mask.sample_name == sample_name:
                    if mask_series is None:
                        mask_series = mask.series
                    else:
                        mask_series = mask_series | mask.series

        return mask_series

    def number_probes_masked(self, mask_name=None, sample_name: str| None=None) -> int:
        """Return the number or masked probes for a specific sample or for all samples if no sample name is provided.

        :return: number of masked probes
        :rtype: int"""
        return sum(self.get_mask(mask_name, sample_name))

    def reset_masks(self):
        """Reset all masks."""
        self.masks = {}

    def remove_masks(self, mask_name=None, sample_name: str| None=None) -> None:
        """Reset the mask for a specific sample or for all samples if no sample name is provided.

        """
        if sample_name is None and mask_name is None:
            self.reset_masks()
        elif sample_name is None:
            self.masks = {k: v for k, v in self.masks.items() if v.mask_name != mask_name}  # remove all masks with the given name
        elif mask_name is None:
            self.masks = {k: v for k, v in self.masks.items() if v.sample_name != sample_name}  # remove all masks for the given sample
        else:
            self.masks.pop((mask_name, sample_name), None)  # remove a specific mask

    # define a copy method
    def copy(self):
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
