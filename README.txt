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
All Python modules used (random, csv, math, time, sys, os) are
part of Python's standard library.  The priority queue is provided
by our own MaxHeap implementation in heap.py — heapq is NOT used.

----------------------------------------------------------------
2. FILE STRUCTURE
----------------------------------------------------------------
group-project/
  main.py               — entry point (launch this file)
  profile.py            — UserProfile data class
  dataset.py            — dataset generation and CSV I/O
  distance.py           — weighted distance function
  heap.py               — custom MaxHeap (replaces heapq)
  baseline.py           — baseline linear-scan search
  kdtree.py             — optimised k-d tree search
  benchmark.py          — experiment and benchmarking script
  README.txt            — this file
  user_profiles.csv     — pre-generated dataset (100,000 profiles)
  user_profiles_1m.csv  — extended dataset (1,000,000 profiles)
  benchmark_results.txt — full experiment results
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

-- f) Full experiment script --
  python3 benchmark.py
 
  Runs 5 experiments comparing Baseline vs K-D Tree across
  different dataset sizes, k values, and weight combinations.
  Results are saved automatically to benchmark_results.txt.

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
7. BENCHMARK RESULTS SUMMARY
----------------------------------------------------------------
Full results are available in benchmark_results.txt.

  Experiment 1 — Dataset Size (k=5, equal weights)
      100,000 profiles: Baseline= 22.778ms, K-D Tree=0.131ms, Speedup=173.8×
      200,000 profiles: Baseline= 46.234ms, K-D Tree=0.103ms, Speedup=448.1×
      300,000 profiles: Baseline= 68.943ms, K-D Tree=0.114ms, Speedup=606.6×
      500,000 profiles: Baseline=116.519ms, K-D Tree=0.117ms, Speedup=998.7×
    1,000,000 profiles: Baseline=(skipped), K-D Tree=0.185ms, Speedup=    —

  Experiment 2 — Value of k (n=100,000, equal weights)
    k=1:  Speedup=380.6×
    k=5:  Speedup=200.4×
    k=10: Speedup=141.7×
    k=15: Speedup=106.0×
    k=20: Speedup= 86.8×

  Experiment 3 — Weight combinations (n=100,000, k=5)
    K-D Tree handles all 6 weight configurations correctly (✓ PASS)

  Experiment 4 — Correctness
    20/20 random queries matched perfectly (100% pass rate)

  Experiment 5 — K-D Tree build time
      100,000 profiles built in   244.55ms
      200,000 profiles built in   432.97ms
      500,000 profiles built in 1,523.72ms
    1,000,000 profiles built in 3,033.43ms (one-off cost)
 
----------------------------------------------------------------
8. BENCHMARK / EXPERIMENT SCRIPT
----------------------------------------------------------------
To run the full set of experiments (Member 4's benchmarking):
 
  python3 benchmark.py
 
  Runs 5 experiments comparing Baseline vs K-D Tree across
  different dataset sizes, k values, and weight combinations.
  Results are saved automatically to benchmark_results.txt.
 
----------------------------------------------------------------
9. TEAM MEMBERS & CONTRIBUTIONS
----------------------------------------------------------------
  Member 1 — Nga Mai Thanh  25%  profile.py, dataset.py, distance.py
                                  Data generation, encoding,
                                  normalisation, distance function
  Member 2 — Nam            25%  baseline.py
                                  Brute-force k-NN search
  Member 3 — Adam           25%  kdtree.py
                                  K-D Tree construction and search
  Member 4 — Yoshita Sarin  25%  benchmark.py, benchmark_results.txt,
                                  README.txt, experiments,
                                  report consolidation
 
----------------------------------------------------------------
10. DEMO VIDEO LINK
----------------------------------------------------------------
  [Insert link here after recording]
 
================================================================