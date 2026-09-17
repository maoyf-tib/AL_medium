
from random import Random
from math import floor, log10
import pandas as pd
from pathlib import Path

'''
Usage Instructions:
Modify the parameters below and run the program.

Example of random feature selection range:
FEATURES = {
    "temperature": {"min": 10, "max": 30, "significant digits": 3},                 
    "concentration": {"min": 0, "max": 1, "endpoints": 10, "decimal digits": 0},    
    "material": {"selection list": ["A", "B", "C"]},                                      
}
# 3 significant digits (default is 4); cannot be used together with decimal_places
# 10 endpoints, 0 decimal places (default is 2)
# Randomly select one from the input list
'''

# Project name, used as the prefix for sample serial numbers; when set to None, no sample serial number is assigned
PROJECT_NAME = None

# Number of samples to output
SAMPLES = 10000

# Storage path and filename for the output samples (must include the .csv extension)
PROJECT_ROOT = Path.cwd()
CSV_PATH = PROJECT_ROOT/'Example'/'Round1'/'output'/'value_combination.csv'

# Random seed (data generated with the same seed is reproducible; 'None' means no random seed is specified)
RANDOMSEED = 42

# Sample features and their random ranges
FEATURES = {
    "(NH4)2SO4": {"min": 0, "max": 68, "endpoints": 10, "decimal digits": 0},
    "Triton X-100": {"min": 0, "max": 53.125, "endpoints": 10, "decimal digits": 0},
    "Glycine": {"min": 0, "max": 85, "endpoints": 10, "decimal digits": 0},
    "CSL-P": {"min": 0, "max": 204, "endpoints": 10, "decimal digits": 0},
    "NaCl": {"min": 17, "max": 51, "endpoints": 10, "decimal digits": 0},
    "K2HPO4": {"min": 17, "max": 51, "endpoints": 10, "decimal digits": 0},
    "Tryptone": {"min": 51, "max": 153, "endpoints": 10, "decimal digits": 0},
    "YE": {"min": 81.6, "max": 244.8, "endpoints": 10, "decimal digits": 0},
    "Methionine": {"min": 0, "max": 204, "endpoints": 10, "decimal digits": 0},
    "Cysteine": {"min": 0, "max": 102, "endpoints": 10, "decimal digits": 0},
    "NH4OAc": {"min": 0, "max": 56.66666667, "endpoints": 10, "decimal digits": 0},
    "Glycerol": {"min": 42.5, "max": 127.5, "endpoints": 10, "decimal digits": 0},
    "Na2S2O3": {"min": 10.625, "max": 42.5, "endpoints": 10, "decimal digits": 0},
    "ZnSO4·7H2O": {"min": 0, "max": 53.125, "endpoints": 10, "decimal digits": 0},
    "MgSO4·7H2O": {"min": 5.3125, "max": 42.5, "endpoints": 10, "decimal digits": 0},
    "FAC": {"min": 6.375, "max": 51, "endpoints": 10, "decimal digits": 0}
}



def generate_samples(samples, features, project_name, seed=None):

    # Initialize the random number generator with a fixed seed
    rng = Random(seed)

    # Generate a random value according to the supplied specification
    def sample(spec):
        if "significant digits" in spec and "decimal digits" in spec:
            raise ValueError("significant digits and decimal digits cannot be set together")

        decimals = spec.get("decimal digits", 2)
        digits = spec.get("significant digits", 4)
        if decimals is not None:
            if not isinstance(decimals, int) or decimals < 0:
                raise ValueError("decimal digits must be a non-negative integer")
        elif not isinstance(digits, int) or digits < 1:
            raise ValueError("significant digits must be a positive integer")

        # Round using either decimal places or significant digits
        def rounded(value):
            if not isinstance(value, (int, float)):
                return value
            if decimals is not None:
                return round(value, decimals)
            if value == 0:
                return value
            return round(value, digits - 1 - floor(log10(abs(value))))

        if "selection list" in spec:
            if not spec["selection list"]:
                raise ValueError("selection list cannot be empty")
            return rounded(rng.choice(spec["selection list"]))

        low, high = spec["min"], spec["max"]
        if low > high:
            raise ValueError("min cannot be greater than max")

        if "endpoints" not in spec:
            return rounded(rng.uniform(low, high))
        parts = spec["endpoints"]
        if not isinstance(parts, int) or parts < 2:
            raise ValueError("endpoints must be an integer greater than or equal to 2")
        return rounded(low + (high - low) * rng.randrange(parts) / (parts - 1))

    # Return each generated sample as a dictionary
    rows = [{**({"sequence": f"{project_name}_{i}"} if project_name is not None else {}),
             **{name: sample(spec) for name, spec in features.items()}}
            for i in range(1, samples + 1)]
    
    return rows



if __name__ == "__main__" :
    rows = generate_samples(SAMPLES, FEATURES, PROJECT_NAME, seed=RANDOMSEED)
    print(rows)
    # Write the generated samples to a CSV file
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
