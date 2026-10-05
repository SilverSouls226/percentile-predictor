import numpy as np

# Load distribution
data = np.load("data/cgpa_distribution.npy")

n = len(data)

while True:
    try:
        cgpa = float(input("Enter CGPA (or -1 to exit): "))
        if cgpa == -1:
            break

        # Binary search
        count = np.searchsorted(data, cgpa, side='right')

        percentile = (count / n) * 100

        print(f"Estimated Percentile: {percentile:.2f}%\n")

    except Exception as e:
        print("Invalid input\n")
