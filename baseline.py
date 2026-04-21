"""
baseline.py - Baseline k-NN search via full dataset scan (linear search).

Algorithm
---------
For each query, iterate over every profile in the dataset and maintain a
max-heap of size k that holds the k smallest distances seen so far.

* If the heap has fewer than k entries, push unconditionally.
* Otherwise, if the current distance is smaller than the largest distance in
  the heap (heap[0]), replace the root.

Using a max-heap of the k nearest candidates lets us determine in O(1) whether
a new candidate could improve the answer.

Complexity
----------
Time  (build): O(n)   — just stores a reference to the profile list
Time  (query): O(n)   — one pass through all n profiles;
                         each heap operation is O(log k), k <= 20, so O(log k)
                         is effectively O(1) here.
Space (query): O(k)   — heap holds at most k entries at any time

No external libraries are used.  Only Python's built-in 'heapq' module is used
for the priority-queue operations.
"""

import heapq

from distance import weighted_distance


class LinearSearch:
    """Full-scan k-nearest-neighbour search."""

    def __init__(self, profiles):
        """
        Parameters
        ----------
        profiles : list[UserProfile]
            The complete dataset.  The list is stored by reference (not copied).
        """
        self.profiles = profiles

    # ----------------------------------------------------------------- search --

    def search(self, query, k, weights):
        """
        Find the k nearest neighbours of *query* by scanning all profiles.

        Parameters
        ----------
        query   : UserProfile
            The query profile (may or may not be in the dataset).
        k       : int
            Number of nearest neighbours to retrieve (1 <= k <= 20).
        weights : list[float]
            Five non-negative attribute weights
            [w_age, w_income, w_degree, w_hours, w_domain].

        Returns
        -------
        list of (float, UserProfile)
            Exactly min(k, n) pairs sorted by distance ascending.
        """
        # Max-heap: each entry is (-distance, profile_id, profile).
        # Negating the distance turns Python's min-heap into a max-heap so that
        # heap[0] always exposes the *largest* distance in the current top-k set.
        # profile_id breaks ties deterministically without comparing profile objects.
        heap = []

        for profile in self.profiles:
            dist = weighted_distance(query, profile, weights)

            if len(heap) < k:
                heapq.heappush(heap, (-dist, profile.id, profile))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, profile.id, profile))

        # Convert to ascending order for output
        # * Primary sort by distance ascending
        # * Secondary sort by profile_id ascending to break ties consistently.
        ordered = sorted(
            ((-neg_d, pid, p) for neg_d, pid, p in heap),
            key=lambda x: (x[0], x[1])
        )
        return [(dist, p) for dist, _, p in ordered]
