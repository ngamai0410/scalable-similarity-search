================================================================
  Scalable Similarity Search System
  RMIT Algorithms & Analysis 2026A — Postgraduate Project
================================================================

----------------------------------------------------------------
1. ENVIRONMENT SETUP
----------------------------------------------------------------
Requirements
  - Python 3.8 or later  (no third-party libraries needed)

Check your Python version:
  python3 --version

No virtual environment or package installation is required.
All modules used (random, csv, heapq, math, time, sys, os)
are part of Python's standard library.

----------------------------------------------------------------
2. FILE STRUCTURE
----------------------------------------------------------------
group-project/
  main.py        — entry point (launch this file)
  profile.py     — UserProfile data class
  dataset.py     — dataset generation and CSV I/O
  distance.py    — weighted distance function
  baseline.py    — baseline linear-scan search
  kdtree.py      — optimised k-d tree search
  README.txt     — this file

  user_profiles.csv  (generated automatically on first run)

----------------------------------------------------------------
3. HOW TO RUN
----------------------------------------------------------------
All commands are run from inside the group-project/ directory.

  cd "path/to/group-project"

-- a) Interactive query mode (default) --
  python3 main.py

  The program will generate (or load) the dataset, build both
  search indices, then prompt you for a query profile, k, and
  attribute weights. Both the baseline and k-d tree results are
  displayed side-by-side with timing information.

-- b) Generate dataset only --
  python3 main.py --generate

  Generates 100 000 profiles and saves them to user_profiles.csv,
  then exits.

-- c) Custom dataset size --
  python3 main.py --size 200000

  Uses 200 000 profiles instead of the default 100 000.
  (The --size flag can be combined with any other flag.)

-- d) Benchmark mode --
  python3 main.py --benchmark

  Runs 10 random queries automatically and prints average query
  times for both the baseline and the k-d tree, plus speedup.

-- e) Demo mode --
  python3 main.py --demo

  Runs a single canned query and shows full output — useful for
  demonstrations and the submission video.

----------------------------------------------------------------
4. INPUT FORMAT
----------------------------------------------------------------
Query profile (prompted interactively):
  age                 integer  18 – 70
  income              integer  5 – 100  (millions VND / month)
  highest_degree      menu choice:
                        1. High School
                        2. Bachelor
                        3. Master
                        4. PhD
  self_learning_hours float    0.0 – 4.0  (hours / day)
  favourite_domain    menu choice:
                        1. AI
                        2. Software Engineering
                        3. Data Science
                        4. Cybersecurity
                        5. Business Analytics

Query parameters:
  k         integer  1 – 20  (number of nearest neighbours)
  weights   5 space-separated floats >= 0, one per attribute
            order:  age  income  degree  hours  domain
            example: 1.0 0.5 1.5 1.0 2.0

----------------------------------------------------------------
5. OUTPUT FORMAT
----------------------------------------------------------------
For each query, the system prints:

  Baseline — Linear Scan  [<time> ms]
    1. dist=<value>  |  Profile #<id>: age=..., income=..., ...
    2. ...
    ...
    k. ...

  Optimised — K-D Tree  [<time> ms]
    (same format)

  Baseline time : X ms
  K-D Tree time : Y ms  (speedup: Z×)
  Correctness   : PASS — both methods return identical neighbours

----------------------------------------------------------------
6. ALGORITHM SUMMARY
----------------------------------------------------------------
Distance function
  Weighted Euclidean distance on normalised attributes.
  Numerical/ordinal attrs normalised to [0, 1].
  Nominal attr (domain) uses binary distance: 0 (same) / 1 (diff).

  d(a, b, w) = sqrt( w0*(Δage)² + w1*(Δincome)²
                   + w2*(Δdegree)² + w3*(Δhours)²
                   + w4*(Δdomain)² )

Baseline (baseline.py)
  Full dataset scan with a max-heap of size k.
  Time:  O(n) per query
  Space: O(k)

K-D Tree (kdtree.py)
  Build:  balanced binary tree, depth ≈ log₂(n).
          Splits cycle through all 5 dimensions.
          Build time: O(n log² n)
  Query:  branch-and-bound with bounding-box pruning.
          Average query time: O(k · n^(1−1/d)) for d=5 dimensions.

----------------------------------------------------------------
7. DEMO VIDEO LINK
----------------------------------------------------------------
  [Insert link here after recording]

================================================================
