# CGPA Percentile Predictor 📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-Data_Science-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A statistical modeling tool that uses **Monte Carlo simulations** to generate realistic university CGPA distributions. It accounts for latent student ability, inter-course correlation, and grading curves to accurately predict what percentile a specific CGPA falls into.

## 🚀 Features
* **Monte Carlo Engine**: Simulates hundreds of thousands of student profiles across multiple courses.
* **Correlated Grading**: Models the statistical probability of consistent academic performance using latent variables.
* **Interactive Predictor**: A fast binary-search CLI tool to instantly look up the exact percentile of any given CGPA.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SilverSouls226/percentile-predictor.git
   cd percentile-predictor
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Generate the Distribution
First, run the simulation engine. This computes the mathematical distribution based on the configured grading boundaries and saves the data to the `data/` directory.

```bash
python generate_distribution.py
```
*Outputs: `data/cgpa_distribution.npy` and a visual histogram in `output/`.*

### 2. Predict a Percentile
Once the distribution is generated, run the interactive predictor to look up percentiles in real-time.

```bash
python predict.py
```
*The script will prompt you to enter a CGPA (e.g., `8.85`). Enter `-1` to exit.*

## 🔬 Experimental History
The `experiments/` directory contains the iterative evolutionary history of the simulation model (from `v1` to `v4`), showing how the latent variable distributions and grade boundary logic were refined over time.
