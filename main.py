"""
main.py - Entry point for the Scalable Similarity Search System.

Usage
-----
    python main.py                        # interactive query mode (default)
    python main.py --generate             # generate & save dataset, then exit
    python main.py --size N               # specify dataset size (default 100 000)
    python main.py --benchmark            # run automated benchmark and exit
    python main.py --demo                 # run a canned demo query and exit

The system:
  1. Generates (or loads) a dataset of user profiles.
  2. Builds a LinearSearch baseline and a KDTree index.
  3. Processes queries — both methods are run for every query so results and
     timings can be compared side-by-side.

No external libraries are used.
"""

import sys
import time

from profile     import UserProfile
from dataset     import DatasetGenerator
from baseline    import LinearSearch
from kdtree      import KDTree


# ======================================================== constants / config ==

DATASET_FILE    = 'user_profiles.csv'
DEFAULT_SIZE    = 100_000
BENCHMARK_RUNS  = 10
BENCHMARK_K     = 10
BENCHMARK_SEED  = 999

SEPARATOR  = '=' * 65
SEPARATOR2 = '-' * 65


# ============================================================= display utils ==

def _header():
    print(SEPARATOR)
    print('  Scalable Similarity Search System')
    print('  RMIT Algorithms & Analysis 2026A  —  Postgraduate Project')
    print(SEPARATOR)


def _print_results(results, label, elapsed_s):
    print(f'\n{SEPARATOR2}')
    print(f'  {label}  [{elapsed_s * 1000:.1f} ms]')
    print(SEPARATOR2)
    for rank, (dist, profile) in enumerate(results, 1):
        print(f'  {rank:>2}. dist={dist:.6f}  |  {profile}')


def _compare(baseline_res, kd_res, baseline_t, kd_t):
    print(f'\n  Baseline time : {baseline_t * 1000:.1f} ms')
    print(f'  K-D Tree time : {kd_t * 1000:.1f} ms', end='')
    if kd_t > 0:
        print(f'  (speedup: {baseline_t / kd_t:.1f}×)', end='')
    print()

    bl_ids = {p.id for _, p in baseline_res}
    kd_ids = {p.id for _, p in kd_res}
    if bl_ids == kd_ids:
        print('  Correctness   : PASS — both methods return identical neighbours')
    else:
        print('  Correctness   : MISMATCH — results differ (investigate!)')


# ========================================================= dataset handling ==

def _load_or_generate(size):
    if DatasetGenerator.exists(DATASET_FILE):
        print(f'\nLoading dataset from "{DATASET_FILE}" …', end='', flush=True)
        t0       = time.time()
        profiles = DatasetGenerator.load(DATASET_FILE)
        elapsed  = time.time() - t0
        print(f' done  ({len(profiles):,} profiles, {elapsed:.2f}s)')
    else:
        print(f'\nDataset not found — generating {size:,} profiles …',
              end='', flush=True)
        t0       = time.time()
        profiles = DatasetGenerator.generate(size)
        elapsed  = time.time() - t0
        print(f' done  ({elapsed:.2f}s)')
        print(f'Saving to "{DATASET_FILE}" …', end='', flush=True)
        t0 = time.time()
        DatasetGenerator.save(profiles, DATASET_FILE)
        print(f' done  ({time.time() - t0:.2f}s)')
    return profiles


def _build_indices(profiles):
    print('\nBuilding search indices …')
    baseline = LinearSearch(profiles)
    print('  [1/2] LinearSearch : ready')

    print('  [2/2] KDTree       : building …', end='', flush=True)
    t0     = time.time()
    kdtree = KDTree(profiles)
    elapsed = time.time() - t0
    print(f' done  ({elapsed:.2f}s)')
    return baseline, kdtree


# ============================================================= query runner ==

def _run_query(query, k, weights, baseline, kdtree):
    print(f'\nQuery  : {query}')
    print(f'k      : {k}')
    print(f'Weights: age={weights[0]}, income={weights[1]}, '
          f'degree={weights[2]}, hours={weights[3]}, domain={weights[4]}')
    print(f'\nSearching …')

    t0 = time.time()
    bl_results = baseline.search(query, k, weights)
    bl_time    = time.time() - t0

    t0 = time.time()
    kd_results = kdtree.search(query, k, weights)
    kd_time    = time.time() - t0

    _print_results(bl_results, 'Baseline — Linear Scan', bl_time)
    _print_results(kd_results, 'Optimised — K-D Tree',   kd_time)
    _compare(bl_results, kd_results, bl_time, kd_time)


# =========================================================== input helpers ==

def _safe_input(prompt_text):
    """Wrapper around input() that handles EOF / KeyboardInterrupt gracefully."""
    try:
        return input(prompt_text)
    except (EOFError, KeyboardInterrupt):
        print('\nExiting.')
        sys.exit(0)


def _get_int(prompt_text, lo, hi):
    while True:
        raw = _safe_input(prompt_text).strip()
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f'  Please enter a value between {lo} and {hi}.')
        except ValueError:
            print('  Invalid input — please enter an integer.')


def _get_float(prompt_text, lo, hi):
    while True:
        raw = _safe_input(prompt_text).strip()
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f'  Please enter a value between {lo} and {hi}.')
        except ValueError:
            print('  Invalid input — please enter a number.')


def _get_choice(prompt_text, options):
    while True:
        raw = _safe_input(prompt_text).strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print(f'  Please enter a number between 1 and {len(options)}.')


def _get_weights():
    print('\nEnter 5 attribute weights  [age  income  degree  hours  domain]')
    print('  All weights must be >= 0.  Example: 1 1 1 1 1')
    while True:
        raw = _safe_input('  Weights: ').strip().split()
        try:
            w = [float(x) for x in raw]
            if len(w) == 5 and all(x >= 0 for x in w):
                return w
            print('  Need exactly 5 non-negative numbers.')
        except ValueError:
            print('  Invalid input — please enter 5 numbers separated by spaces.')


# ============================================================ interactive mode ==

def _interactive_mode(baseline, kdtree):
    print(f'\n{SEPARATOR}')
    print('  Interactive Query Mode  (press Ctrl-C or type "quit" to exit)')
    print(SEPARATOR)

    while True:
        # ---- Build query profile ----
        print('\n--- Query Profile ---')
        age = _get_int(
            f'  Age ({UserProfile.AGE_MIN}–{UserProfile.AGE_MAX}): ',
            UserProfile.AGE_MIN, UserProfile.AGE_MAX)

        income = _get_int(
            f'  Income in millions VND ({UserProfile.INCOME_MIN}–{UserProfile.INCOME_MAX}): ',
            UserProfile.INCOME_MIN, UserProfile.INCOME_MAX)

        print('  Degree options:')
        for i, d in enumerate(UserProfile.DEGREES, 1):
            print(f'    {i}. {d}')
        degree = _get_choice('  Select degree (1–4): ', UserProfile.DEGREES)

        hours = _get_float(
            f'  Self-learning hours/day (0–{UserProfile.HOURS_MAX}): ',
            0.0, UserProfile.HOURS_MAX)

        print('  Domain options:')
        for i, d in enumerate(UserProfile.DOMAINS, 1):
            print(f'    {i}. {d}')
        domain = _get_choice('  Select domain (1–5): ', UserProfile.DOMAINS)

        query = UserProfile(-1, age, income, degree, hours, domain)

        # ---- Query parameters ----
        k       = _get_int('\nNumber of neighbours k (1–20): ', 1, 20)
        weights = _get_weights()

        _run_query(query, k, weights, baseline, kdtree)

        print()
        again = _safe_input('\nRun another query? (y/n): ').strip().lower()
        if again != 'y':
            break


# ============================================================= benchmark mode ==

def _benchmark_mode(profiles, baseline, kdtree):
    import random
    random.seed(BENCHMARK_SEED)

    print(f'\n{SEPARATOR}')
    print('  Benchmark Mode')
    print(SEPARATOR)
    print(f'  {BENCHMARK_RUNS} random queries,  k={BENCHMARK_K},  '
          f'weights=[1, 1, 1, 1, 1]')
    print()

    weights = [1.0] * 5
    k       = BENCHMARK_K

    total_bl = 0.0
    total_kd = 0.0

    for i in range(BENCHMARK_RUNS):
        query = random.choice(profiles)

        t0 = time.time()
        bl = baseline.search(query, k, weights)
        total_bl += time.time() - t0

        t0 = time.time()
        kd = kdtree.search(query, k, weights)
        total_kd += time.time() - t0

        bl_ids = {p.id for _, p in bl}
        kd_ids = {p.id for _, p in kd}
        status = 'OK' if bl_ids == kd_ids else 'MISMATCH'
        print(f'  Query {i + 1:2d}: status={status}')

    avg_bl = total_bl / BENCHMARK_RUNS * 1000
    avg_kd = total_kd / BENCHMARK_RUNS * 1000

    print(f'\n  Average baseline time : {avg_bl:.2f} ms')
    print(f'  Average K-D Tree time : {avg_kd:.2f} ms')
    if avg_kd > 0:
        print(f'  Average speedup       : {avg_bl / avg_kd:.1f}×')


# ================================================================= demo mode ==

def _demo_mode(profiles, baseline, kdtree):
    """Run a canned demo query — useful for the submission video."""
    print(f'\n{SEPARATOR}')
    print('  Demo Mode')
    print(SEPARATOR)

    query = UserProfile(
        profile_id        = -1,
        age               = 28,
        income            = 30,
        highest_degree    = 'Master',
        self_learning_hours = 2.5,
        favourite_domain  = 'AI',
    )
    k       = 5
    weights = [1.0, 0.5, 1.5, 1.0, 2.0]   # emphasise domain and degree

    _run_query(query, k, weights, baseline, kdtree)


# ================================================================== main =====

def _parse_args():
    args      = sys.argv[1:]
    mode      = 'interactive'
    size      = DEFAULT_SIZE

    i = 0
    while i < len(args):
        a = args[i]
        if a == '--generate':
            mode = 'generate'
        elif a == '--benchmark':
            mode = 'benchmark'
        elif a == '--demo':
            mode = 'demo'
        elif a == '--size' and i + 1 < len(args):
            i += 1
            try:
                size = int(args[i])
                if size < 1:
                    raise ValueError
            except ValueError:
                print(f'Invalid --size value: {args[i]}')
                sys.exit(1)
        elif a in ('--help', '-h'):
            print(__doc__)
            sys.exit(0)
        else:
            print(f'Unknown argument: {a}')
            sys.exit(1)
        i += 1

    return mode, size


def main():
    _header()
    mode, size = _parse_args()

    profiles = _load_or_generate(size)

    if mode == 'generate':
        print('\nDataset ready.  Exiting.')
        return

    baseline, kdtree = _build_indices(profiles)

    if mode == 'benchmark':
        _benchmark_mode(profiles, baseline, kdtree)
    elif mode == 'demo':
        _demo_mode(profiles, baseline, kdtree)
    else:
        _interactive_mode(baseline, kdtree)

    print('\nDone.')


if __name__ == '__main__':
    main()
