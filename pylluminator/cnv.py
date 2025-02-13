"""Functions to compute the Copy Number Variation (CNV), ie detect changes in the number of copies on a particular
region of the genome, and splits the genome in segments with similar CNV.
"""

import pandas as pd
import pyranges as pr
import numpy as np
import linear_segment
from sklearn.linear_model import LinearRegression

from pylluminator.samples import Samples, read_samples, from_sesame
from pylluminator.annotations import ArrayType, Annotations, PYLLUMINA_DATA_LINK
from pylluminator.utils import get_resource_folder, download_from_geo, download_from_link
from pylluminator.utils import get_logger

LOGGER = get_logger()


def copy_number_variation(samples: Samples, sample_label: str, normalization_sample_labels: str | list[str] | None = None) -> (pr.PyRanges, pd.DataFrame, pd.DataFrame):
    """Perform copy number variation (CNV) and copy number segmentation for a sample

    :param samples: samples to be analyzed
    :type samples: Samples

    :param sample_label: name of the samples to calculate CNV of.
    :type sample_label: str

    :param normalization_sample_labels: names of the samples to use for normalization, that have to be part of the passed
        Samples object. If empty (default), default normalization samples will be loaded from a database - but this only
        work for EPIC/hg38 and EPICv2/hg38; for other array versions, you **need** to provide samples. Default []
    :type normalization_sample_labels: str | list[str]

    :return: a tuple with : the bins coordinates, the bins signal, the segments
    :rtype: tuple[pyranges.PyRanges, pandas.DataFrame, pandas.DataFrame]
    """

    available_samples = samples.sample_labels
    if sample_label not in available_samples:
        LOGGER.error(f'Sample {sample_label} not found in the Samples object')
        return None

    if normalization_sample_labels is not None:

        # make sure it's a list
        if isinstance(normalization_sample_labels, str):
            normalization_sample_labels = [normalization_sample_labels]

        # extract normalization samples from the samples object.
        for norm_sample_name in normalization_sample_labels:
            if norm_sample_name not in available_samples:
                LOGGER.error(f'Normalization sample {norm_sample_name} not found in the Samples object')
        normalization_sample_labels = [n for n in normalization_sample_labels if n in available_samples]

        if len(normalization_sample_labels) == 0:
            LOGGER.error('None of the normalization samples were found in the Samples object, exiting')
            return None

        norm_intensities = samples.get_total_ib_intensity()[normalization_sample_labels]

    else:

        # load default normalization samples for the given array version
        normal_samples = get_normalization_samples(samples.annotation)

        if normal_samples is None:
            print('Please provide samples to use as normalization')
            return None

        norm_intensities = normal_samples.get_total_ib_intensity()

    genome_info = samples.annotation.genome_info
    probe_coords_df = samples.annotation.genomic_ranges

    # get total intensity per probe and drop unnecessary indexes
    target_intensity = samples.get_total_ib_intensity(sample_label)
    target_intensity = target_intensity.droplevel(['channel', 'type', 'probe_type']).dropna()

    norm_intensities = norm_intensities.droplevel(['channel', 'type', 'probe_type']).dropna()

    # keep only probes that are in all 3 files (target methylation, normalization methylation and genome ranges)
    overlapping_probes = target_intensity.index.intersection(norm_intensities.index).intersection(probe_coords_df.index)
    LOGGER.debug(f'Keeping {len(overlapping_probes)} overlapping probes')
    target_intensity = target_intensity.loc[overlapping_probes]
    norm_intensities = norm_intensities.loc[overlapping_probes]
    probe_coords_df = probe_coords_df.loc[overlapping_probes]

    LOGGER.info('Fitting the linear regression on normalization intensities')

    X = norm_intensities.values
    y = target_intensity.values
    fitted_model = LinearRegression().fit(X, y)
    predicted = np.maximum(fitted_model.predict(X), 1)
    probe_coords_df['cnv'] = np.log2(target_intensity / predicted)

    # make tiles
    tile_width = 50000
    tiles = pr.tile_genome(genome_info.seq_length, tile_width).reset_index(drop=True).sort_ranges()
    diff_tiles = tiles.subtract_ranges(genome_info.gap_info).reset_index(drop=True)

    # make bins
    non_empty_coords = probe_coords_df[probe_coords_df.end > probe_coords_df.start]  # remove 0-width ranges
    probe_coords = pr.PyRanges(non_empty_coords.rename(columns={'chromosome':'Chromosome', 'end': 'End', 'start': 'Start',
                                                          'strand': 'Strand'}))
    diff_tiles = diff_tiles.count_overlaps(probe_coords.reset_index())

    if len(diff_tiles) == 0:
        LOGGER.error('No diff tiles')
        return None, None, None

    # merge small bins together, until they reach a minimum of 20 overlapping probes #todo optimize (10sec)
    bins = _merge_bins_to_minimum_overlap(diff_tiles, probe_coords, 20, 1)

    # segment the signal
    if len(bins) == 0:
        LOGGER.error('No bins')
        return None, None, None

    joined_pr = probe_coords.reset_index().join_ranges(bins, suffix='_bin')
    signal_bins = joined_pr.groupby(['Chromosome', 'Start_bin', 'End_bin'])['cnv'].median().reset_index()
    signal_bins = signal_bins.rename(columns={'Chromosome': 'chromosome', 'Start_bin': 'start_bin', 'End_bin': 'end_bin'})
    signal_bins['map_loc'] = ((signal_bins['start_bin'] + signal_bins['end_bin']) / 2).astype(int)

    # todo : improve this method (and optimize : 15sec)
    cn_seg = linear_segment.segment(signal_bins.cnv.values.astype('double'),
                                    labels=signal_bins.chromosome.astype(str).values,
                                    method='cbs', shuffles=10000, p=0.0001)

    # merge the segmentation information with the signal info
    df_seg = pd.DataFrame(zip(cn_seg.starts, cn_seg.ends, cn_seg.labels), columns=['start', 'end', 'chromosome'])
    seg_values = []

    for chromosome in set(signal_bins.chromosome):
        chrom_df = signal_bins[signal_bins.chromosome == chromosome].reset_index()
        chrom_segs = df_seg[df_seg.chromosome == chromosome]
        for seg_id, seg_value in chrom_segs.iterrows():
            start_pos = chrom_df.loc[seg_value.start].map_loc
            end_pos = chrom_df.loc[seg_value.end - 1].map_loc
            nb_bins = len(chrom_df.loc[seg_value.start:seg_value.end - 1])
            mean_cnv = chrom_df.loc[seg_value.start:seg_value.end - 1].cnv.mean()
            seg_values.append([chromosome, seg_id, start_pos, end_pos, nb_bins, mean_cnv])

    seg_df = pd.DataFrame(seg_values, columns=['chromosome', 'seg_id', 'start', 'end', 'nb_bins', 'mean_cnv'])

    return probe_coords, signal_bins, seg_df.set_index('seg_id')


def get_normalization_samples(annotation: Annotations) -> Samples | None:
    """Read from the package's data normalization samples data, depending on the array type.

    Only EPIC v2 and EPIC are supported for now.

    :param annotation: annotation used for your samples
    :type annotation: Annotations

    :return: samples to use for normalization, corresponding to the given annotation type and version, or None if no
        normalization samples exist for this annotation
    :rtype: Samples | None"""
    LOGGER.info('Getting normalization samples')

    if annotation.array_type == ArrayType.HUMAN_EPIC_V2:

        idat_dir = get_resource_folder('arrays.epic_v2_normalization_data')
        gsm_ids = ['GSM7139626', 'GSM7139627']
        download_from_geo(gsm_ids, idat_dir)
        return read_samples(idat_dir, annotation=annotation)

    if annotation.array_type == ArrayType.HUMAN_EPIC:

        datadir = get_resource_folder('arrays.epic_normalization_data')
        files = ['NAF.A.csv.gz', 'NAF.B.csv.gz', 'NAF.C.csv.gz', 'PrEC.Rep1.csv.gz', 'PrEC.Rep2.csv.gz']
        missing_files = [f for f in files if not datadir.joinpath(f).exists()]
        for missing_file in missing_files:
            download_from_link(f'{PYLLUMINA_DATA_LINK}/arrays/epic_normalization_data/{missing_file}', datadir)
        return from_sesame(datadir, annotation)

    LOGGER.error(f'No predefined normalization data for array {annotation.array_type}')
    return None


def _merge_bins_to_minimum_overlap(pr_to_merge: pr.PyRanges, pr_to_overlap_with: pr.PyRanges, minimum_overlap: int = 20,
                                  precision:float = 1) -> pr.PyRanges:
    """Merge adjacent intervals from `pr_to_merge` until they have a minimum probes overlap such as defined in parameter
    `minimum_overlap`.


    Overlap count is calculated with `pr_to_overlap_with`.

    :param pr_to_merge: intervals to merge
    :type pr_to_merge: pyranges.PyRanges

    :param pr_to_overlap_with: intervals used as reference for overlap calculation
    :type pr_to_overlap_with: pyranges.PyRanges

    :param minimum_overlap: target minimum overlap for each bin. Merging of adjacent intervals stops when this number
        is reached. Default: 20
    :type minimum_overlap: int

    :param precision: must be between 0 and minimum_overlap. 0 is the maximum precision, meaning that resulting
        intervals will be on average smaller (closer to the minimum) - but it comes at a cost : a higher computing time.
        Default: 1
    :type precision: float

    :return: intervals with probes overlap >= `minimum_overlap`
    :rtype: pyranges.PyRanges"""

    pr_to_merge = pr_to_merge.reset_index(drop=True)  # to ensure an int type index, not object
    pr_to_overlap_with = pr_to_overlap_with.reset_index(drop=True)  # to ensure an int type index, not object
    columns_ini = pr_to_merge.columns  # to restore columns at the end

    # we can't already have a 'Cluster' column as clustering will fail if so
    if 'Cluster' in columns_ini:
        columns_ini = columns_ini.drop('Cluster')

    # count overlaps with other pyRanges object
    if 'NumberOverlaps' not in columns_ini:
        pr_to_merge = pr_to_merge.count_overlaps(pr_to_overlap_with)

    # for best precision, do an extra step of merging only small tiles together
    if precision == 0:
        for current_min in [n for n in range(pr_to_merge.NumberOverlaps.min(), minimum_overlap)]:
            has_high_overlap = pr_to_merge.NumberOverlaps > current_min
            high_overlaps = pr_to_merge[has_high_overlap]
            low_overlaps = pr_to_merge[~has_high_overlap]
            low_overlaps_merged = low_overlaps.merge_overlaps(slack=1).count_overlaps(pr_to_overlap_with)
            pr_to_merge = pr.concat([low_overlaps_merged, high_overlaps]).sort_ranges()

    # main merge technique : see iteratively if ranges with low number of overlapping probes have neighbors they can
    # merge with. Iterations are useful for a more precise result - the bigger the steps, the bigger the intervals

    precision = np.clip(precision, a_min=1, a_max=minimum_overlap)  # ensure precision is within bounds
    # we need to force the value of minimum overlap in the iteration loop, as it can be excluded depending on precision
    mins = [n for n in range(max(1, pr_to_merge.NumberOverlaps.min()), minimum_overlap, precision)] + [minimum_overlap]

    for current_min in mins:
        pr_to_merge = pr_to_merge.sort_ranges().cluster(slack=1)  # cluster intervals to identify neighbors
        needs_merge = pr_to_merge.NumberOverlaps < current_min
        is_left_neighbor_in_cluster = pr_to_merge.Cluster.diff() == 0
        is_right_neighbor_in_cluster = pr_to_merge.Cluster.diff(-1) == 0

        to_merge_left = needs_merge & is_left_neighbor_in_cluster  # low overlap rows that are going to be merged
        to_merge_left |= to_merge_left.shift(-1, fill_value=False)  # neighbors that are going to merge

        to_merge_right = needs_merge & is_right_neighbor_in_cluster & ~to_merge_left  # do not merge right if merge left
        to_merge_right |= to_merge_right.shift(fill_value=False)

        to_merge = to_merge_right | to_merge_left  # merge indexes of identified intervals together
        if not to_merge.any():
            LOGGER.debug(f'Nothing to merge for min {current_min}')
            pr_to_merge = pr_to_merge[columns_ini]
            continue

        # finally, merge identified intervals and update interval overlap
        merged_bins = pr_to_merge[to_merge].merge_overlaps(slack=1).count_overlaps(pr_to_overlap_with)
        # add intervals that were not involved in the merge step and filter out unnecessary columns
        pr_to_merge = pr.concat([merged_bins, pr_to_merge[~to_merge]]).reset_index()[columns_ini]

    # only return rows that have a number of overlaps above (>=) threshold
    return pr_to_merge[pr_to_merge.NumberOverlaps >= minimum_overlap].sort_ranges()
