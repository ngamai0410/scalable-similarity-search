"""
kdtree.py - K-D Tree for optimised k-nearest-neighbour search.

Algorithm overview
------------------
Build phase
~~~~~~~~~~~
  1. Choose the splitting axis by cycling through dimensions 0–4
     (axis = depth mod 5).
  2. Sort the current profile subset on that axis and pick the median element
     as the node.
  3. Recurse on the left half (< median index) and the right half (> median
     index), incrementing depth each time.

  This produces a balanced binary tree of height ≈ log₂(n).

  Build complexity:  O(n log² n)  - sorting O(n log n) at each of log n levels.
  Space complexity:  O(n)         - one node per profile.

Search phase (branch-and-bound)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The search maintains a max-heap of the k best candidates found so far, then
  prunes entire subtrees when their bounding box is provably farther than the
  current k-th distance.

  The bounding box of each subtree is tracked implicitly: when we recurse into
  a child, we update the relevant dimension of the parent's bounding box and
  restore it afterward (backtracking), avoiding extra memory allocations.

  Lower-bound distance to a bounding box
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For continuous dimensions 0–3:
        If q[i] < bmin[i]  →  min_diff = bmin[i] - q[i]
        If q[i] > bmax[i]  →  min_diff = q[i]  - bmax[i]
        Otherwise          →  min_diff = 0
        contribution       = weights[i] * min_diff²

    For the nominal domain dimension 4:
        The bounding box tracks [min_domain_idx, max_domain_idx].
        If the query's domain index falls within this integer range, a profile
        with the same domain *may* exist in the subtree → min contribution = 0.
        Otherwise every profile in the subtree has a different domain
        → min contribution = weights[4] * 1.0.

  Search complexity:  O(k · n^(1−1/d)) average for d dimensions.
                      O(n) worst case (degrades to linear scan in high-d
                      spaces - known as the curse of dimensionality for d≥5).

Dimension details
-----------------
  Axis  Attribute              Range in vector
  ----  --------------------   ---------------
   0    age                    [0, 1]  (continuous, normalised)
   1    income                 [0, 1]  (continuous, normalised)
   2    highest_degree         [0, 1]  (ordinal:  0, 1/3, 2/3, 1)
   3    self_learning_hours    [0, 1]  (continuous, normalised)
   4    favourite_domain       {0,1,2,3,4}  (raw index; binary dist used)

No external libraries are used.  The priority queue is provided by our own
MaxHeap implementation in heap.py.
"""

import math
import sys

from distance import weighted_distance
from heap     import MaxHeap


# ================================================================ Node class ==

class _KDNode:
    """Internal tree node.  Uses __slots__ to reduce per-object memory."""

    __slots__ = ('profile', 'axis', 'split_val', 'left', 'right')

    def __init__(self, profile, axis):
        self.profile   = profile
        self.axis      = axis
        self.split_val = profile.vector[axis]
        self.left      = None
        self.right     = None


# ================================================================= KDTree ===

class KDTree:
    """
    K-D Tree supporting exact k-nearest-neighbour queries with branch-and-bound
    pruning.

    Parameters
    ----------
    profiles : list[UserProfile]
        The dataset.  The list is copied internally so the caller may modify
        theirs without affecting the tree.
    """

    NDIM        = 5          # number of vector dimensions
    DOMAIN_MIN  = 0.0        # minimum raw domain index stored in dim 4
    DOMAIN_MAX  = 4.0        # maximum raw domain index stored in dim 4

    def __init__(self, profiles):
        # Raise Python's recursion limit to accommodate large datasets safely.
        # For n=100 000 the balanced-tree height is ≈17, well below 1 000,
        # but we set 10 000 as a conservative guard.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
        self._root = self._build(list(profiles), depth=0)

    # ============================================================== Build ====

    def _build(self, profiles, depth):
        """
        Recursively build the k-d tree.

        Parameters
        ----------
        profiles : list[UserProfile]   (will be sorted in place at each level)
        depth    : int

        Returns
        -------
        _KDNode or None
        """
        n = len(profiles)
        if n == 0:
            return None

        axis = depth % self.NDIM

        # Sort on the splitting axis and choose the median as pivot.
        profiles.sort(key=lambda p: p.vector[axis])
        mid = n // 2

        node       = _KDNode(profiles[mid], axis)
        node.left  = self._build(profiles[:mid],      depth + 1)
        node.right = self._build(profiles[mid + 1:],  depth + 1)
        return node

    # ============================================================== Search ====

    def search(self, query, k, weights):
        """
        Find the k nearest neighbours of *query* using branch-and-bound.

        Parameters
        ----------
        query   : UserProfile
        k       : int   (1 <= k <= 20)
        weights : list[float]   five non-negative weights

        Returns
        -------
        list of (float, UserProfile)
            Exactly min(k, dataset_size) pairs, sorted by distance ascending.
        """
        q_vec = query.vector
        q_domain = query.domain_idx

        # Max-heap of size k.  Stores (distance, profile_id, profile); the
        # largest distance is at the root and represents the current "worst"
        # of the top-k candidates.  profile_id breaks ties deterministically.
        heap = MaxHeap()

        # We track the bounding box implicitly by modifying bmin/bmax in place
        # and restoring them after each recursive call (backtracking).
        bmin = [0.0, 0.0, 0.0, 0.0, self.DOMAIN_MIN]
        bmax = [1.0, 1.0, 1.0, 1.0, self.DOMAIN_MAX]

        def _min_bbox_dist(bmin, bmax):
            """
            Minimum possible weighted distance from *query* to any point
            inside the axis-aligned bounding box [bmin, bmax].
            """
            sq = 0.0

            # Continuous dimensions 0–3
            for i in range(4):
                qi = q_vec[i]
                lo = bmin[i]
                hi = bmax[i]
                if qi < lo:
                    diff = lo - qi
                elif qi > hi:
                    diff = qi - hi
                else:
                    diff = 0.0
                sq += weights[i] * diff * diff

            # Nominal dimension 4 (binary distance)
            if not (bmin[4] <= q_domain <= bmax[4]):
                # No profile in this subtree can share the query's domain
                sq += weights[4]  # * 1.0²

            return math.sqrt(sq)

        def _search(node):
            if node is None:
                return

            # ---- Pruning ------------------------------------------------
            # Once we have k candidates, prune this subtree if its closest
            # possible point is no better than the current k-th distance.
            if len(heap) == k and _min_bbox_dist(bmin, bmax) >= heap.peek()[0]:
                return

            # ---- Visit current node ------------------------------------
            dist = weighted_distance(query, node.profile, weights)
            if len(heap) < k:
                heap.push((dist, node.profile.id, node.profile))
            elif dist < heap.peek()[0]:
                heap.replace_root((dist, node.profile.id, node.profile))

            # ---- Recurse into children ---------------------------------
            axis = node.axis
            sv = node.split_val
            diff = q_vec[axis] - sv

            # Visit the nearer half-space first for better pruning.
            if diff <= 0:
                # Query is on the left side: explore left child first.
                # Left child's bounding box: cap bmax[axis] at sv.
                old = bmax[axis]
                bmax[axis] = sv
                _search(node.left)
                bmax[axis] = old

                # Right child's bounding box: raise bmin[axis] to sv.
                old = bmin[axis]
                bmin[axis] = sv
                _search(node.right)
                bmin[axis] = old
            else:
                # Query is on the right side: explore right child first.
                old = bmin[axis]
                bmin[axis] = sv
                _search(node.right)
                bmin[axis] = old

                old = bmax[axis]
                bmax[axis] = sv
                _search(node.left)
                bmax[axis] = old

        _search(self._root)

        # Return sorted ascending by distance with deterministic tie-break on id
        ordered = sorted(heap.items(), key=lambda x: (x[0], x[1]))
        return [(dist, p) for dist, _, p in ordered]
