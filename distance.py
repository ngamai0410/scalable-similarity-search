"""Weighted Euclidean distance between two UserProfile instances."""

import math


def weighted_distance(profile_a, profile_b, weights):
    """Return weighted Euclidean distance between two profiles."""
    va = profile_a.vector
    vb = profile_b.vector

    # Dimensions 0–3: normalised continuous difference
    d0 = va[0] - vb[0]
    d1 = va[1] - vb[1]
    d2 = va[2] - vb[2]
    d3 = va[3] - vb[3]

    dist_sq = (weights[0] * d0 * d0
             + weights[1] * d1 * d1
             + weights[2] * d2 * d2
             + weights[3] * d3 * d3)

    # Dimension 4: binary distance for nominal favourite_domain
    if profile_a.domain_idx != profile_b.domain_idx:
        dist_sq += weights[4]   # * 1.0^2

    return math.sqrt(dist_sq)
