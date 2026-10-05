import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# CONFIGURATION
# -----------------------

NUM_STUDENTS = 200000
NUM_COURSES = 31

# Your stats (WITHOUT F)
mu = 7.727594086
sigma = 0.5377751267

your_cgpa = 8.85

# Try multiple correlation strengths
alpha_values = [0.2, 0.4, 0.6, 0.8]

# -----------------------
# GRADE BOUNDARIES
# -----------------------

def get_grade_points(x):
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
        return 5  # No F now

get_grade_points_vec = np.vectorize(get_grade_points)

# -----------------------
# SIMULATION LOOP
# -----------------------

for alpha in alpha_values:

    # Latent ability
    A = np.random.normal(0, 1, NUM_STUDENTS)

    # Noise
    epsilon = np.random.normal(0, 1, (NUM_STUDENTS, NUM_COURSES))

    # Generate scores
    X = mu + sigma * (alpha * A[:, None] + np.sqrt(1 - alpha**2) * epsilon)

    # Convert to grade points
    grades = get_grade_points_vec(X)

    # CGPA
    cgpa = grades.mean(axis=1)

    # Percentile
    percentile = np.mean(cgpa <= your_cgpa) * 100

    # Some diagnostics
    print(f"\nAlpha = {alpha}")
    print(f"Estimated Percentile: {percentile:.2f}%")
    print(f"CGPA Mean: {cgpa.mean():.3f}")
    print(f"CGPA SD: {cgpa.std():.3f}")

plt.hist(cgpa, bins=50)
plt.axvline(your_cgpa)
plt.show()
