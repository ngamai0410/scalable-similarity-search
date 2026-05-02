"""
heap.py - Custom binary MaxHeap implemented from scratch.

A MaxHeap is a complete binary tree stored in an array such that the value
at every parent node is greater than or equal to the values at its children.
This invariant lets us:
  * inspect the largest element in O(1) time (peek),
  * insert a new element in O(log n) time (push),
  * replace the largest element with a new one in O(log n) time
    (replace_root) without an intermediate pop step.

In this project, the heap stores tuples of the form
    (distance, profile_id, profile)
ordered by Python's natural tuple comparison.  The largest distance is
therefore always at the root, which represents the current "worst" of the
top-k candidates.  When a new candidate has a smaller distance than the
root, we replace the root and sift down - this is exactly what we need for
bounded top-k retrieval.

Tie-breaking on profile_id is handled implicitly by tuple comparison: if two
candidates share the same distance, the one with the larger profile_id
floats to the root.  Since profile_ids are unique, the comparison never has
to fall through to the third tuple element (the profile object itself).

Time complexity
---------------
push           : O(log n)
peek           : O(1)
replace_root   : O(log n)
items / clear  : O(n)

Space complexity
----------------
The heap stores n elements in a single Python list.
"""


class MaxHeap:
    """Binary max-heap backed by a Python list."""

    __slots__ = ("_items",)

    def __init__(self):
        self._items = []

    # ------------------------------------------------------------------ size --

    def __len__(self):
        return len(self._items)

    def is_empty(self):
        return not self._items

    # ----------------------------------------------------------------- access --

    def peek(self):
        """Return the largest element without removing it.  O(1)."""
        return self._items[0]

    def items(self):
        """Return a shallow copy of the underlying array (unordered)."""
        return list(self._items)

    # --------------------------------------------------------------- mutation --

    def push(self, item):
        """Insert *item* into the heap.  O(log n)."""
        self._items.append(item)
        self._sift_up(len(self._items) - 1)

    def replace_root(self, item):
        """
        Replace the root with *item* and restore the heap property.

        Equivalent to a pop followed by a push but in a single sift-down
        pass - this is the operation we use to swap a worse candidate out
        for a better one when maintaining the top-k set.
        """
        self._items[0] = item
        self._sift_down(0)

    def clear(self):
        self._items.clear()

    # --------------------------------------------------------------- internals --

    def _sift_up(self, idx):
        """Move the element at *idx* up toward the root until the max-heap
        invariant is restored."""
        items = self._items
        while idx > 0:
            parent = (idx - 1) >> 1
            if items[idx] > items[parent]:
                items[idx], items[parent] = items[parent], items[idx]
                idx = parent
            else:
                return

    def _sift_down(self, idx):
        """Move the element at *idx* down toward the leaves until the
        max-heap invariant is restored."""
        items = self._items
        n = len(items)
        while True:
            left = (idx << 1) + 1
            right = left + 1
            largest = idx
            if left < n and items[left] > items[largest]:
                largest = left
            if right < n and items[right] > items[largest]:
                largest = right
            if largest == idx:
                return
            items[idx], items[largest] = items[largest], items[idx]
            idx = largest
