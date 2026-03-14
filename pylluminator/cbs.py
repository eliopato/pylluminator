"""Circular Binary Segmentation (CBS) for change point detection in 1D signals.

Reimplements the CBS algorithm from the linear_segment library (which is unmaintained
and incompatible with numpy>=2.0) using an O(n) CUSUM-based statistic with vectorized
permutation testing. The original C implementation used the same O(n) approach but
required Cython/C compilation; this version achieves comparable performance through
numpy vectorization.

Reference: Venkatraman, E.S. and Olshen, A.B., 2007. A faster circular binary
segmentation algorithm for the analysis of array CGH data. Bioinformatics, 23(6),
pp.657-663.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SegmentationResult:
    """Result of CBS segmentation, compatible with linear_segment's interface.

    Attributes hold parallel lists: the i-th element of each list describes one segment.
    Indices (starts/ends) are 0-based within each label group.
    """
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def _cbs_stat(x: np.ndarray) -> tuple[float, int, int]:
    """Find the most anomalous segment using O(n) CUSUM min/max approach.

    Single-pass algorithm matching the linear_segment C implementation: computes
    cumulative sum of centered data, identifies the segment between positions of
    the minimum and maximum CUSUM values.

    :param x: 1D signal array
    :return: (t_statistic, segment_start, segment_end)
    """
    n = len(x)
    if n < 4:
        return 0.0, 0, n

    cumsum = np.cumsum(x - np.mean(x))

    i_max = int(np.argmax(cumsum))
    i_min = int(np.argmin(cumsum))

    i0 = min(i_max, i_min)
    i1 = max(i_max, i_min)

    seg_start = i0 + 1
    seg_end = i1 + 1
    seg_len = seg_end - seg_start

    if seg_len == 0 or seg_len == n:
        return 0.0, 0, n

    diff = cumsum[i1] - cumsum[i0]
    denom = float(seg_len) * float(n + 1 - i1 + i0)
    if denom <= 0:
        return 0.0, 0, n

    t_stat = float(diff ** 2 * n / denom)
    return t_stat, seg_start, seg_end


def _vectorized_cbs_t_stats(perms: np.ndarray) -> np.ndarray:
    """Compute CBS t-statistics for a batch of permutations.

    Vectorized O(n) per row — processes all permutations simultaneously using
    the same CUSUM min/max approach as _cbs_stat.

    :param perms: 2D array of shape (batch_size, n) with permuted data
    :return: 1D array of t-statistics, one per permutation
    """
    batch_size, n = perms.shape
    means = perms.mean(axis=1, keepdims=True)
    cumsums = np.cumsum(perms - means, axis=1)

    i_maxs = np.argmax(cumsums, axis=1)
    i_mins = np.argmin(cumsums, axis=1)

    i0s = np.minimum(i_maxs, i_mins)
    i1s = np.maximum(i_maxs, i_mins)

    idx = np.arange(batch_size)
    diffs = cumsums[idx, i1s] - cumsums[idx, i0s]
    seg_lens = (i1s - i0s).astype(np.float64)

    t_stats = np.zeros(batch_size, dtype=np.float64)
    valid = seg_lens > 0
    if valid.any():
        denoms = seg_lens[valid] * (n + 1 - i1s[valid] + i0s[valid]).astype(np.float64)
        nonzero = denoms > 0
        if nonzero.any():
            valid_idx = np.where(valid)[0][nonzero]
            t_stats[valid_idx] = diffs[valid_idx] ** 2 * n / denoms[nonzero]

    return t_stats


def _permutation_test(x: np.ndarray, observed_t: float, shuffles: int,
                      p: float, rng: np.random.Generator,
                      batch_size: int = 500) -> bool:
    """Batch-vectorized permutation test using the CBS t-statistic.

    Uses the same O(n) statistic as _cbs_stat for consistency — the original
    linear_segment C code also uses calculate_cbs_stat for both observed and
    permuted data. Stops early once enough exceedances are found.

    :param x: original 1D signal
    :param observed_t: the CBS t-statistic of the observed data
    :param shuffles: number of permutations
    :param p: significance threshold
    :param rng: numpy random generator
    :param batch_size: permutations per vectorized batch
    :return: True if the observed statistic is significant at level p
    """
    n = len(x)
    exceed_count = 0
    threshold = int(p * shuffles) + 1

    for batch_start in range(0, shuffles, batch_size):
        current_batch = min(batch_size, shuffles - batch_start)
        perms = np.empty((current_batch, n), dtype=np.float64)
        for i in range(current_batch):
            perms[i] = rng.permutation(x)

        t_stats = _vectorized_cbs_t_stats(perms)
        exceed_count += int(np.sum(t_stats >= observed_t))

        if exceed_count >= threshold:
            return False

    return True


def _cbs_test(x: np.ndarray, shuffles: int, p: float,
              rng: np.random.Generator) -> tuple[bool, int, int]:
    """Test for a significant change point using CBS.

    Computes the CBS t-statistic and runs a permutation test to determine
    significance. Applies edge-snapping heuristics from the original C
    implementation: boundaries within 5 of an edge are snapped to that edge,
    and segments shorter than 5 are rejected.

    :param x: 1D signal array
    :param shuffles: number of permutations for significance testing
    :param p: p-value threshold
    :param rng: numpy random generator
    :return: (is_significant, segment_start, segment_end)
    """
    n = len(x)
    if n < 4:
        return False, 0, n

    t_stat, start, end = _cbs_stat(x)

    if t_stat == 0.0:
        return False, 0, n

    if start < 5:
        start = 0
    if n - end < 5:
        end = n

    seg_len = end - start
    if seg_len < 5 or seg_len == n:
        return False, 0, n

    significant = _permutation_test(x, t_stat, shuffles, p, rng)
    return significant, start, end


def _recursive_cbs(x: np.ndarray, offset: int, shuffles: int, p: float,
                   rng: np.random.Generator,
                   segments: list[tuple[int, int]], min_size: int = 5) -> None:
    """Recursively apply CBS to find all change points.

    Splits at the most significant segment and recurses on the resulting
    sub-segments until no more significant change points are found.

    :param x: 1D signal array (sub-segment of the original)
    :param offset: index offset of x within the label group
    :param shuffles: number of permutations
    :param p: significance threshold
    :param rng: numpy random generator
    :param segments: accumulator list of (start, end) tuples
    :param min_size: minimum segment size to attempt splitting
    """
    if len(x) < min_size:
        segments.append((offset, offset + len(x)))
        return

    significant, start, end = _cbs_test(x, shuffles, p, rng)

    if not significant:
        segments.append((offset, offset + len(x)))
        return

    if start > 0:
        _recursive_cbs(x[:start], offset, shuffles, p, rng, segments, min_size)

    _recursive_cbs(x[start:end], offset + start, shuffles, p, rng, segments, min_size)

    if end < len(x):
        _recursive_cbs(x[end:], offset + end, shuffles, p, rng, segments, min_size)


def segment(x: np.ndarray, labels: np.ndarray,
            method: str = 'cbs', shuffles: int = 10000,
            p: float = 0.0001) -> SegmentationResult:
    """Segment a 1D signal using Circular Binary Segmentation.

    Drop-in replacement for linear_segment.segment(). Segmentation is performed
    independently within each contiguous group of identical labels.

    :param x: 1D array of signal values (e.g. log2 CNV ratios)
    :param labels: array of group labels (e.g. chromosome names), same length as x
    :param method: segmentation method, only 'cbs' is supported
    :param shuffles: number of permutations for the significance test
    :param p: p-value threshold for declaring a change point significant
    :return: SegmentationResult with starts, ends, and labels for each segment
    :raises ValueError: if method is not 'cbs' or array lengths don't match
    """
    if method != 'cbs':
        raise ValueError(f"Only 'cbs' method is supported, got '{method}'")

    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)

    if len(x) != len(labels):
        raise ValueError(f"x and labels must have the same length, got {len(x)} and {len(labels)}")

    result = SegmentationResult()
    rng = np.random.default_rng()

    unique_labels: list[str] = []
    seen: set[str] = set()
    for lbl in labels:
        lbl_str = str(lbl)
        if lbl_str not in seen:
            unique_labels.append(lbl_str)
            seen.add(lbl_str)

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        group_x = x[indices]

        segments: list[tuple[int, int]] = []
        _recursive_cbs(group_x, 0, shuffles, p, rng, segments)
        segments.sort()

        for seg_start, seg_end in segments:
            result.starts.append(seg_start)
            result.ends.append(seg_end)
            result.labels.append(label)

    return result
