import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# -----------------------
# CONFIG
# -----------------------

NUM_RUNS = 10
NUM_STUDENTS = 100000
NUM_COURSES = 31

your_cgpa = 8.85

TRAITS = 4

# -----------------------
# STORAGE
# -----------------------

results = []
all_cgpas = []

# -----------------------
# SIMULATION FUNCTION
# -----------------------

def run_simulation():

    students = np.random.normal(0, 1, (NUM_STUDENTS, TRAITS))

    course_weights = np.random.uniform(0.2, 1.0, (NUM_COURSES, TRAITS))
    course_weights /= course_weights.sum(axis=1, keepdims=True)

    course_difficulty = np.random.normal(0, 0.5, NUM_COURSES)

    grades = np.zeros((NUM_STUDENTS, NUM_COURSES))

    for j in range(NUM_COURSES):

        ability = students @ course_weights[j]
        noise = np.random.normal(0, 0.5, NUM_STUDENTS)

        raw = ability - course_difficulty[j] + noise

        mu = np.mean(raw)
        sigma = np.std(raw)

        grades[:, j] = np.where(
            raw >= mu + 1.5 * sigma, 10,
            np.where(raw >= mu + 0.5 * sigma, 9,
            np.where(raw >= mu - 0.5 * sigma, 8,
            np.where(raw >= mu - 1.5 * sigma, 7,
            np.where(raw >= mu - 2.5 * sigma, 6, 5)))))
    
    cgpa = grades.mean(axis=1)

    percentile = np.mean(cgpa <= your_cgpa) * 100
    sd = np.std(cgpa)

    return cgpa, percentile, sd

# -----------------------
# RUN MULTIPLE TIMES
# -----------------------

for i in range(NUM_RUNS):
    cgpa, percentile, sd = run_simulation()

    results.append([i+1, percentile, sd])
    all_cgpas.append(cgpa)

# -----------------------
# TABLE OUTPUT (ENHANCED)
# -----------------------

percentiles = [r[1] for r in results]
sds = [r[2] for r in results]

avg_percentile = np.mean(percentiles)
std_percentile = np.std(percentiles)

avg_sd = np.mean(sds)
std_sd = np.std(sds)

cv_percentile = std_percentile / avg_percentile
cv_sd = std_sd / avg_sd

# Add per-run CV (optional: relative SD of CGPA)
enhanced_results = []

for i, (run, perc, sd) in enumerate(results[:-1]):
    enhanced_results.append([
        run,
        perc,
        sd,
        sd / avg_sd  # relative SD vs mean SD
    ])

# Final summary row
enhanced_results.append([
    "AVG",
    avg_percentile,
    avg_sd,
    "-"
])

enhanced_results.append([
    "STD",
    std_percentile,
    std_sd,
    "-"
])

enhanced_results.append([
    "CV",
    cv_percentile,
    cv_sd,
    "-"
])

print(tabulate(
    enhanced_results,
    headers=["Run", "Percentile", "CGPA SD", "Rel SD"],
    floatfmt=".4f"
))
# -----------------------
# OVERLAY GRAPH
# -----------------------

plt.figure(figsize=(8,5))

for cgpa in all_cgpas:
    plt.hist(cgpa, bins=50, alpha=0.1)

plt.axvline(your_cgpa)

plt.title("Overlay of Multiple Simulations")
plt.xlabel("CGPA")
plt.ylabel("Frequency")

plt.show()

# -----------------------
# AVERAGE DISTRIBUTION
# -----------------------

combined = np.concatenate(all_cgpas)

plt.figure(figsize=(8,5))
plt.hist(combined, bins=50)

plt.axvline(your_cgpa)

plt.title("Final Averaged CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")

plt.show()
