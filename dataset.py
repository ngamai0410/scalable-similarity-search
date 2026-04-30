"""Dataset generation and persistence using only built-in random and csv modules."""

import csv
import os
import random

from profile import (UserProfile, DEGREES, DOMAINS,
                     AGE_MIN, AGE_MAX, INCOME_MIN, INCOME_MAX, HOURS_MAX)


class DatasetGenerator:
    DEFAULT_SIZE = 100_000
    DEFAULT_FILE = 'user_profiles.csv'

    _HEADER = ['id', 'age', 'income', 'highest_degree',
               'self_learning_hours', 'favourite_domain']

    @staticmethod
    def generate(size=DEFAULT_SIZE, seed=42):
        """Generate *size* random UserProfile objects with a fixed seed."""
        random.seed(seed)

        degrees = DEGREES
        domains = DOMAINS
        age_min, age_max = AGE_MIN, AGE_MAX
        inc_min, inc_max = INCOME_MIN, INCOME_MAX
        hours_max        = HOURS_MAX

        profiles = []
        for i in range(size):
            age    = random.randint(age_min, age_max)
            income = random.randint(inc_min, inc_max)
            degree = random.choice(degrees)
            hours  = round(random.uniform(0.0, hours_max), 2)
            domain = random.choice(domains)
            profiles.append(UserProfile(i, age, income, degree, hours, domain))

        return profiles

    # ------------------------------------------------------------------- Save --

    @staticmethod
    def save(profiles, filepath=DEFAULT_FILE):
        """Write profiles to a CSV file."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(DatasetGenerator._HEADER)
            for p in profiles:
                writer.writerow(p.to_row())

    # ------------------------------------------------------------------- Load --

    @staticmethod
    def load(filepath=DEFAULT_FILE):
        """Load profiles from a previously saved CSV file."""
        profiles = []
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profiles.append(UserProfile(
                    int(row['id']),
                    int(row['age']),
                    int(row['income']),
                    row['highest_degree'],
                    float(row['self_learning_hours']),
                    row['favourite_domain'],
                ))
        return profiles

    # ----------------------------------------------------------------- Helper --

    @staticmethod
    def exists(filepath=DEFAULT_FILE):
        return os.path.isfile(filepath)
