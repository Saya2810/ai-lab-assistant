import random
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "app"),
)

from services.rag_engine.vectorstore import (
    index_experiment,
)

random.seed(42)

samples = ["GaAs", "AlGaAs", "InGaAs"]
temperatures = [4, 10, 20, 50, 77, 100, 150, 300]
laser_powers = [0.1, 0.5, 1, 2, 5, 10]

for i in range(1, 51):

    material = random.choice(samples)
    temperature = random.choice(temperatures)
    laser_power = random.choice(laser_powers)

    peak = round(random.uniform(700, 950), 2)
    intensity = round(random.uniform(100, 10000), 2)

    experiment = {
        "id": f"test-{i}",
        "title": f"PL measurement of {material} sample {i}",
        "date": "2026-11-06",
        "body": (
            f"SYNTHETIC TEST DATA — NOT A REAL EXPERIMENT.\n"
            f"Material: {material}\n"
            f"Temperature: {temperature} K\n"
            f"Laser power: {laser_power} mW\n"
            f"Emission peak: {peak} nm\n"
            f"PL intensity: {intensity} counts\n"
            f"The photoluminescence spectrum was "
            f"recorded using a CCD spectrometer."
        ),
    }

    index_experiment(experiment)

print("50 synthetic experiments indexed.")
