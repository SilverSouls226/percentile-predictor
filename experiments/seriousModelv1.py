import numpy as np
import matplotlib.pyplot as plt
# -----------------------
# CONFIG
# -----------------------

NUM_STUDENTS = 100000
NUM_COURSES = 31

your_cgpa = 8.85

# Traits
TRAITS = ["reasoning", "memory", "consistency", "speed"]
NUM_TRAITS = len(TRAITS)

# -----------------------
# GENERATE STUDENTS
# -----------------------

# Each student has a vector of traits
students = np.random.normal(0, 1, (NUM_STUDENTS, NUM_TRAITS))

# -----------------------
# GENERATE COURSES
# -----------------------

# Each course has weights for each trait
course_weights = np.random.uniform(0.2, 1.0, (NUM_COURSES, NUM_TRAITS))

# Normalize weights so sum = 1
course_weights = course_weights / course_weights.sum(axis=1, keepdims=True)

# Course difficulty
course_difficulty = np.random.normal(0, 0.5, NUM_COURSES)

# -----------------------
# GRADE FUNCTION
# -----------------------

def get_grade_points(x, mu, sigma):
    if x >= mu + 1.5 * sigma:
        return 10
    elif x >= mu + 0.5 * sigma:
        return 9
    elif x >= mu - 0.5 * sigma:
        return 8
    elif x >= mu - 1.5 * sigma:
        return 7
    elif x >= mu - 2.5 * sigma:
        return 6
    else:
        return 5

# -----------------------
# SIMULATION
# -----------------------

grades = np.zeros((NUM_STUDENTS, NUM_COURSES))

for j in range(NUM_COURSES):

    # Weighted ability
    ability_score = students @ course_weights[j]

    # Add difficulty and noise
    noise = np.random.normal(0, 0.5, NUM_STUDENTS)
    raw_score = ability_score - course_difficulty[j] + noise

    # Normalize to grading scale
    mu = np.mean(raw_score)
    sigma = np.std(raw_score)

    grades[:, j] = np.vectorize(get_grade_points)(raw_score, mu, sigma)

# CGPA
cgpa = grades.mean(axis=1)

# Percentile
percentile = np.mean(cgpa <= your_cgpa) * 100

print(f"Estimated Percentile: {percentile:.2f}%")
print(f"CGPA Mean: {cgpa.mean():.3f}")
print(f"CGPA SD: {cgpa.std():.3f}")


plt.figure(figsize=(8, 5))

# Histogram
counts, bins, patches = plt.hist(cgpa, bins=50)

# Highlight region below your CGPA
for i in range(len(patches)):
    if bins[i] <= your_cgpa:
        patches[i].set_alpha(0.7)
    else:
        patches[i].set_alpha(0.3)

# Vertical line
plt.axvline(your_cgpa)

plt.title("CGPA Distribution with Your Position Highlighted")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")

plt.show()
