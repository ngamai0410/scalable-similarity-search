"""
profile.py - UserProfile data class for the Scalable Similarity Search System.

Each profile stores 5 attributes and precomputes a normalized 5-dimensional vector
used by both search algorithms.

Vector layout (all values in [0, 1] except domain which keeps its raw index):
  [0] age              normalized to [0, 1]  (18–70)
  [1] income           normalized to [0, 1]  (5–100 million VND)
  [2] highest_degree   ordinal, normalized    (High School=0, Bachelor=1/3, Master=2/3, PhD=1)
  [3] self_learning_hours  normalized to [0, 1]  (0–4 h/day)
  [4] favourite_domain raw integer index     (0–4, for tree-splitting only)

The domain attribute uses binary distance in actual distance computation
(0 if identical, 1 if different), not continuous difference.
"""


class UserProfile:
    DEGREES = ['High School', 'Bachelor', 'Master', 'PhD']
    DOMAINS = ['AI', 'Software Engineering', 'Data Science', 'Cybersecurity', 'Business Analytics']

    AGE_MIN, AGE_MAX       = 18, 70
    INCOME_MIN, INCOME_MAX = 5, 100
    HOURS_MAX              = 4.0

    _AGE_RANGE    = AGE_MAX - AGE_MIN          # 52
    _INCOME_RANGE = INCOME_MAX - INCOME_MIN    # 95
    _DEGREE_RANGE = len(DEGREES) - 1           # 3

    def __init__(self, profile_id, age, income, highest_degree,
                 self_learning_hours, favourite_domain):
        self.id                  = profile_id
        self.age                 = age
        self.income              = income
        self.highest_degree      = highest_degree
        self.self_learning_hours = self_learning_hours
        self.favourite_domain    = favourite_domain

        # Precomputed indices for fast access
        self.domain_idx = self.DOMAINS.index(favourite_domain)
        degree_idx      = self.DEGREES.index(highest_degree)

        # Normalized 5-D vector (stored as tuple for immutability and speed)
        self.vector = (
            (age - self.AGE_MIN)          / self._AGE_RANGE,
            (income - self.INCOME_MIN)    / self._INCOME_RANGE,
            degree_idx                    / self._DEGREE_RANGE,
            self_learning_hours           / self.HOURS_MAX,
            float(self.domain_idx),        # raw index used only for tree splitting
        )

    # ------------------------------------------------------------------ I/O --

    def to_row(self):
        """Return a list of values suitable for CSV writing."""
        return [self.id, self.age, self.income, self.highest_degree,
                self.self_learning_hours, self.favourite_domain]

    def __repr__(self):
        return (
            f"Profile #{self.id}: "
            f"age={self.age}, "
            f"income={self.income}M VND, "
            f"degree={self.highest_degree}, "
            f"hours={self.self_learning_hours:.2f}/day, "
            f"domain={self.favourite_domain}"
        )
