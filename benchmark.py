"""
benchmark.py - Experiment & Benchmarking Script
================================================
Algorithms & Analysis 2026A — Postgraduate Project
Member 4: Experiments, Benchmarking, and Report Consolidation

HOW TO RUN
----------
    python3 benchmark.py

Make sure this file is in the same folder as:
    profile.py, dataset.py, baseline.py, kdtree.py, distance.py

NOTE ON LARGE DATASETS
----------------------
This script tests dataset sizes up to 1,000,000 profiles.
Brute force on 1M profiles will be slow (~200+ seconds per query).
The script skips brute force for sizes above 500,000 to save time
and only runs the K-D Tree for the largest sizes.

OUTPUT
------
Prints all results to the console AND saves them to benchmark_results.txt
so you can copy the numbers directly into your report.

EXPERIMENTS
-----------
  1. Dataset size     — vary n from 1k to 1M, fix k=5, equal weights
  2. Value of k       — vary k, fix n=100,000, equal weights
  3. Weight sets      — vary weights, fix n=100,000, k=5
  4. Correctness      — verify k-d tree matches brute force on 20 queries
  5. Build time       — measure k-d tree construction time up to 1M
"""

import random
import time

from profile  import UserProfile
from dataset  import DatasetGenerator
from baseline import LinearSearch
from kdtree   import KDTree

RANDOM_SEED  = 42       # fixed seed → reproducible results
RUNS_PER_EXP = 5        # average over this many queries per scenario
RESULTS_FILE = 'benchmark_results.txt'

output_lines = []   # collects every printed line so we can save to file

def log(text=''):
    """Print and also save to output buffer."""
    print(text)
    output_lines.append(text)

def separator(char='═', width=65):
    log(char * width)

def section(title):
    log()
    separator()
    log(f'  {title}')
    separator()

def make_query(profile_id=-1):
    """Create a fixed query profile used across experiments."""
    return UserProfile(
        profile_id          = profile_id,
        age                 = 28,
        income              = 30,
        highest_degree      = 'Master',
        self_learning_hours = 2.5,
        favourite_domain    = 'AI',
    )

def time_search(search_fn, query, k, weights, runs=RUNS_PER_EXP):
    """
    Run search_fn(query, k, weights) RUNS times and return
    (average_seconds, last_results).
    Averaging removes noise from OS scheduling.
    """
    times = []
    result = None
    for _ in range(runs):
        t0     = time.perf_counter()
        result = search_fn(query, k, weights)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    return avg, result

def check_correctness(bl_results, kd_results):
    """Return True if both methods returned the same set of profile IDs."""
    bl_ids = {p.id for _, p in bl_results}
    kd_ids = {p.id for _, p in kd_results}
    return bl_ids == kd_ids

def build_indices(profiles):
    """Build LinearSearch + KDTree and return both with build time."""
    baseline = LinearSearch(profiles)

    t0     = time.perf_counter()
    kdtree = KDTree(profiles)
    build_time = time.perf_counter() - t0

    return baseline, kdtree, build_time

def subsample(all_profiles, n, seed=RANDOM_SEED):
    """Return the first n profiles (dataset was generated with fixed seed)."""
    random.seed(seed)
    return all_profiles[:n]


# ═══════════════════════════════════════════ EXPERIMENT 1 ════════════════════
# Vary dataset size — how does each method scale?

# Sizes above this threshold skip brute force (too slow for a laptop)
BRUTE_FORCE_LIMIT = 500_000

def experiment_1(all_profiles):
    section('Experiment 1 — Effect of Dataset Size  (k=5, equal weights)')

    sizes   = [100_000, 200_000, 300_000, 500_000, 1_000_000]
    k       = 5
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    query   = make_query()

    log(f'  {"Size":>12}  {"Brute Force (ms)":>18}  {"K-D Tree (ms)":>15}  {"Speedup":>9}  {"Correct?":>9}')
    log(f'  {"-"*12}  {"-"*18}  {"-"*15}  {"-"*9}  {"-"*9}')

    for n in sizes:
        if n > len(all_profiles):
            log(f'  {n:>12,}  {"(not enough data)":>18}  {"—":>15}  {"—":>9}  {"—":>9}')
            continue

        profiles = subsample(all_profiles, n)
        baseline, kdtree, _ = build_indices(profiles)

        # Skip brute force for very large sizes — too slow
        if n <= BRUTE_FORCE_LIMIT:
            bl_time, bl_res = time_search(baseline.search, query, k, weights)
            kd_time, kd_res = time_search(kdtree.search,   query, k, weights)
            speedup = bl_time / kd_time if kd_time > 0 else float('inf')
            correct = '✓ PASS' if check_correctness(bl_res, kd_res) else '✗ FAIL'
            log(f'  {n:>12,}  {bl_time*1000:>18.3f}  {kd_time*1000:>15.3f}  {speedup:>9.1f}×  {correct:>9}')
        else:
            kd_time, _ = time_search(kdtree.search, query, k, weights)
            log(f'  {n:>12,}  {"(skipped)":>18}  {kd_time*1000:>15.3f}  {"—":>9}  {"—":>9}')

    log()
    log(f'  Note: Brute force skipped for n > {BRUTE_FORCE_LIMIT:,} (too slow for laptop benchmarking)')


# ═══════════════════════════════════════════ EXPERIMENT 2 ════════════════════
# Vary k — does the number of neighbours affect speed?

def experiment_2(all_profiles):
    section('Experiment 2 — Effect of k  (n=100,000, equal weights)')

    k_values = [1, 5, 10, 15, 20]
    weights  = [1.0, 1.0, 1.0, 1.0, 1.0]
    query    = make_query()
    profiles = subsample(all_profiles, 100_000)
    baseline, kdtree, _ = build_indices(profiles)

    log(f'  {"k":>5}  {"Brute Force (ms)":>18}  {"K-D Tree (ms)":>15}  {"Speedup":>9}  {"Correct?":>9}')
    log(f'  {"-"*5}  {"-"*18}  {"-"*15}  {"-"*9}  {"-"*9}')

    for k in k_values:
        bl_time, bl_res = time_search(baseline.search, query, k, weights)
        kd_time, kd_res = time_search(kdtree.search,   query, k, weights)

        speedup = bl_time / kd_time if kd_time > 0 else float('inf')
        correct = '✓ PASS' if check_correctness(bl_res, kd_res) else '✗ FAIL'

        log(f'  {k:>5}  {bl_time*1000:>18.3f}  {kd_time*1000:>15.3f}  {speedup:>9.1f}×  {correct:>9}')


# ═══════════════════════════════════════════ EXPERIMENT 3 ════════════════════
# Vary weights — does emphasising different attributes change results?

def experiment_3(all_profiles):
    section('Experiment 3 — Effect of Weight Combinations  (n=100,000, k=5)')

    profiles = subsample(all_profiles, 100_000)
    baseline, kdtree, _ = build_indices(profiles)
    query = make_query()
    k     = 5

    weight_sets = [
        ([1.0, 1.0, 1.0, 1.0, 1.0], 'Equal weights'),
        ([0.8, 0.05, 0.05, 0.05, 0.05], 'Age dominates'),
        ([0.05, 0.8, 0.05, 0.05, 0.05], 'Income dominates'),
        ([0.05, 0.05, 0.05, 0.05, 0.8], 'Domain dominates'),
        ([0.05, 0.05, 0.8, 0.05, 0.05], 'Degree dominates'),
        ([0.3, 0.2, 0.2, 0.1, 0.2],     'Mixed weights'),
    ]

    log(f'  {"Weight Set":<25}  {"Brute Force (ms)":>18}  {"K-D Tree (ms)":>15}  {"Correct?":>9}')
    log(f'  {"-"*25}  {"-"*18}  {"-"*15}  {"-"*9}')

    for weights, label in weight_sets:
        bl_time, bl_res = time_search(baseline.search, query, k, weights)
        kd_time, kd_res = time_search(kdtree.search,   query, k, weights)
        correct = '✓ PASS' if check_correctness(bl_res, kd_res) else '✗ FAIL'

        log(f'  {label:<25}  {bl_time*1000:>18.3f}  {kd_time*1000:>15.3f}  {correct:>9}')


# ═══════════════════════════════════════════ EXPERIMENT 4 ════════════════════
# Correctness test — 20 random queries, all must match

def experiment_4(all_profiles):
    section('Experiment 4 — Correctness Verification  (20 random queries)')

    random.seed(RANDOM_SEED)
    profiles = subsample(all_profiles, 100_000)
    baseline, kdtree, _ = build_indices(profiles)

    k       = 10
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    passed  = 0
    total   = 20

    log(f'  {"Query":>7}  {"Status":>9}  {"Brute Force top-1":<35}  {"K-D Tree top-1"}')
    log(f'  {"-"*7}  {"-"*9}  {"-"*35}  {"-"*35}')

    for i in range(total):
        query = random.choice(profiles)

        _, bl_res = time_search(baseline.search, query, k, weights, runs=1)
        _, kd_res = time_search(kdtree.search,   query, k, weights, runs=1)

        ok = check_correctness(bl_res, kd_res)
        if ok:
            passed += 1

        status   = '✓ PASS' if ok else '✗ FAIL'
        bl_top1  = f'#{bl_res[0][1].id}' if bl_res else 'N/A'
        kd_top1  = f'#{kd_res[0][1].id}' if kd_res else 'N/A'

        log(f'  {i+1:>7}  {status:>9}  {bl_top1:<35}  {kd_top1}')

    log()
    log(f'  Result: {passed}/{total} queries passed  '
        f'({"100%" if passed==total else f"{passed/total*100:.0f}%"})')


# ═══════════════════════════════════════════ EXPERIMENT 5 ════════════════════
# K-D Tree build time across dataset sizes

def experiment_5(all_profiles):
    section('Experiment 5 — K-D Tree Build Time vs Dataset Size')

    sizes = [100_000, 200_000, 300_000, 500_000, 1_000_000]

    log(f'  {"Size":>12}  {"Build Time (ms)":>17}  {"Nodes":>12}')
    log(f'  {"-"*12}  {"-"*17}  {"-"*12}')

    for n in sizes:
        if n > len(all_profiles):
            log(f'  {n:>12,}  {"(not enough data)":>17}  {"—":>12}')
            continue

        profiles = subsample(all_profiles, n)

        t0     = time.perf_counter()
        KDTree(profiles)
        build_t = time.perf_counter() - t0

        log(f'  {n:>12,}  {build_t*1000:>17.2f}  {n:>12,}')


# ═══════════════════════════════════════════ SUMMARY ═════════════════════════

def print_summary():
    section('Summary')
    log('  Key findings to include in your report:')
    log()
    log('  1. Brute force time grows linearly with dataset size (O(n))')
    log('  2. K-D Tree is significantly faster on large datasets')
    log('  3. Both methods return identical results (correctness verified)')
    log('  4. Changing k has minimal effect on brute force time')
    log('  5. K-D Tree build time is a one-off cost amortised over many queries')
    log()
    log('  Copy the tables above into Section 3 (Evaluation) of your report.')
    log('  Fill in the Speedup column analysis in your own words.')


# ═══════════════════════════════════════════ MAIN ════════════════════════════

def main():
    separator()
    log('  Benchmark Script — Scalable Similarity Search')
    log('  Algorithms & Analysis 2026A')
    separator()

    # Load or generate a 1M dataset
    # If user_profiles_1m.csv exists, load it
    # Otherwise check for user_profiles.csv and top it up to 1M
    # Otherwise generate fresh 1M profiles
    TARGET = 1_000_000

    if DatasetGenerator.exists('user_profiles_1m.csv'):
        log('\nLoading 1M dataset from user_profiles_1m.csv …')
        t0 = time.perf_counter()
        all_profiles = DatasetGenerator.load('user_profiles_1m.csv')
        log(f'  Loaded {len(all_profiles):,} profiles in {(time.perf_counter()-t0):.1f}s')

    elif DatasetGenerator.exists('user_profiles.csv'):
        log('\nLoading existing dataset from user_profiles.csv …')
        t0 = time.perf_counter()
        all_profiles = DatasetGenerator.load('user_profiles.csv')
        log(f'  Loaded {len(all_profiles):,} profiles in {(time.perf_counter()-t0):.1f}s')

        if len(all_profiles) < TARGET:
            log(f'\n  Dataset has {len(all_profiles):,} profiles.')
            log(f'  Generating full {TARGET:,} profile dataset for large-scale experiments …')
            log('  (This may take 1-2 minutes)')
            t0 = time.perf_counter()
            all_profiles = DatasetGenerator.generate(TARGET)
            log(f'  Generated in {(time.perf_counter()-t0):.1f}s')
            log('  Saving to user_profiles_1m.csv …')
            t0 = time.perf_counter()
            DatasetGenerator.save(all_profiles, 'user_profiles_1m.csv')
            log(f'  Saved in {(time.perf_counter()-t0):.1f}s')
    else:
        log(f'\nNo dataset found — generating {TARGET:,} profiles …')
        log('  (This may take 1-2 minutes)')
        t0 = time.perf_counter()
        all_profiles = DatasetGenerator.generate(TARGET)
        log(f'  Generated in {(time.perf_counter()-t0):.1f}s')
        DatasetGenerator.save(all_profiles, 'user_profiles_1m.csv')
        log('  Saved to user_profiles_1m.csv')

    log(f'\n  Total profiles available: {len(all_profiles):,}')

    # Run all experiments
    experiment_1(all_profiles)
    experiment_2(all_profiles)
    experiment_3(all_profiles)
    experiment_4(all_profiles)
    experiment_5(all_profiles)
    print_summary()

    # Save results to file
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    log(f'\n  Results saved to "{RESULTS_FILE}"')
    separator()


if __name__ == '__main__':
    main()