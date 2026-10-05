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

# Target F rate (approx)
FAIL_SIGMA_THRESHOLD = 3.5  # corresponds to your earlier grading rule

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
        noise = np.random.normal(0, 0.6, NUM_STUDENTS)  # slightly higher noise to allow F tail

        raw = ability - course_difficulty[j] + noise

        mu = np.mean(raw)
        sigma = np.std(raw)

        grades[:, j] = np.where(
            raw >= mu + 1.5 * sigma, 10,
            np.where(raw >= mu + 0.5 * sigma, 9,
            np.where(raw >= mu - 0.5 * sigma, 8,
            np.where(raw >= mu - 1.5 * sigma, 7,
            np.where(raw >= mu - 2.5 * sigma, 6,
            np.where(raw >= mu - FAIL_SIGMA_THRESHOLD * sigma, 5, 0))))))
    
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
# TABLE OUTPUT
# -----------------------

percentiles = [r[1] for r in results]
sds = [r[2] for r in results]

avg_percentile = np.mean(percentiles)
std_percentile = np.std(percentiles)

avg_sd = np.mean(sds)
std_sd = np.std(sds)

cv_percentile = std_percentile / avg_percentile
cv_sd = std_sd / avg_sd

enhanced_results = []

for i, (run, perc, sd) in enumerate(results):
    enhanced_results.append([run, perc, sd, sd / avg_sd])

enhanced_results.append(["AVG", avg_percentile, avg_sd, "-"])
enhanced_results.append(["STD", std_percentile, std_sd, "-"])
enhanced_results.append(["CV", cv_percentile, cv_sd, "-"])

print(tabulate(
    enhanced_results,
    headers=["Run", "Percentile", "CGPA SD", "Rel SD"],
    floatfmt=".4f"
))

# -----------------------
# SAVE DISTRIBUTION (OVERWRITE OLD FILE)
# -----------------------

combined = np.concatenate(all_cgpas)
combined_sorted = np.sort(combined)

np.save("cgpa_distribution.npy", combined_sorted)

print("Updated distribution saved (with F grades)")

# -----------------------
# SAVE GRAPH
# -----------------------

plt.figure(figsize=(8,5))
plt.hist(combined, bins=50)

plt.axvline(your_cgpa)

plt.title("CGPA Distribution (With F Grades)")
plt.xlabel("CGPA")
plt.ylabel("Frequency")

plt.savefig("cgpa_distribution_with_F.png", dpi=300)
plt.close()

print("Graph saved as cgpa_distribution_with_F.png")
