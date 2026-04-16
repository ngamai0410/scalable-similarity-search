"""
distance.py - Weighted distance function for user profile similarity.

Design decisions
----------------
* Numerical attributes (age, income, self_learning_hours) and the ordinal
  attribute (highest_degree) are normalised to [0, 1] before the distance is
  computed, so each attribute contributes comparably regardless of its raw scale.
* The nominal attribute (favourite_domain) uses binary distance:
      0.0  — profiles share the same domain
      1.0  — profiles belong to different domains
  This avoids imposing a false ordinal structure on an unordered category.
* The final distance is a weighted Euclidean norm:

      d(a, b, w) = sqrt( w0*(Δage)²  + w1*(Δincome)²
                       + w2*(Δdegree)² + w3*(Δhours)²
                       + w4*(Δdomain)² )

  where each Δ is the normalised difference (or binary flag for domain).
  All terms are non-negative, so the result is always >= 0.

No external libraries are used — only the built-in 'math' module.
"""

import math


def weighted_distance(profile_a, profile_b, weights):
    """
    Compute the weighted distance between two UserProfile instances.

    Parameters
    ----------
    profile_a, profile_b : UserProfile
        The two profiles to compare. Both must have a precomputed `.vector`
        tuple and a `.domain_idx` integer attribute.
    weights : sequence of 5 floats
        [w_age, w_income, w_degree, w_hours, w_domain]
        Each weight must be >= 0.

    Returns
    -------
    float
        Non-negative distance value. Returns 0.0 when both profiles are
        identical under the given weights.

    Complexity
    ----------
    Time : O(1)  — fixed five-dimensional computation
    Space: O(1)
    """
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
