from typing import Optional

import kneed


def highly_cited_threshold(
    citation_counts: list[int], top_percentile: Optional[float] = None, min_count: Optional[int] = None
) -> Optional[int]:
    """
    Calculate the threshold for this specific set of citation counts,
    above which the count is likely to be percieved as "highly cited".
    This function first uses the Elbow method.
    If no turning point is found, it returns counts in the given top percentile, including.
    Only counts above the given `min_count` are considered.
    @param citation_counts: list of citation counts, if source papers not given
    @param top_percentile: percentile of counts to consider if no elbow is found, default 0.85
    @param min_count: minimum count to consider, default 10 (from bibliometric study)
    @return: threshold value of counts, *not* the index, or None if not found
    """
    top_percentile = top_percentile or 0.85
    min_count = min_count or 10
    # If no citation counts are given, return None
    if not citation_counts:
        return None
    # If only one citation count is given, return it if it's above the minimum
    if len(citation_counts) == 1:
        if citation_counts[0] >= min_count:
            return citation_counts[0]
        else:
            return None
    sorted_counts = sorted(citation_counts)
    kneedle = kneed.KneeLocator(range(len(sorted_counts)), sorted_counts, curve="convex", direction="increasing")
    # If a knee is found and the threshold is above the minimum, return it
    if kneedle.knee:
        threshold = sorted_counts[kneedle.knee]
        if threshold >= min_count:
            return threshold
    # If no knee is found or knee too low, return the count at the given percentile, if above the minimum
    threshold = sorted_counts[int(len(sorted_counts) * top_percentile)]
    if threshold >= min_count:
        return threshold
    # If no valid threshold is found, return None
    return None
