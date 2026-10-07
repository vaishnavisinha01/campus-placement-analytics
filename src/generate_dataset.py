import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------
np.random.seed(42)

# Number of students
N = 1000


# ---------------------------------------------------------
# 2. GENERATE STUDENT FEATURES
# ---------------------------------------------------------

student_id = [f"S{i:04d}" for i in range(1, N + 1)]

cgpa = np.round(
    np.random.normal(7.5, 1.0, N).clip(5.0, 10.0),
    2
)

dsa_problems = np.random.poisson(120, N).clip(0, 350)

projects = np.random.poisson(2, N).clip(0, 6)

internships = np.random.poisson(0.7, N).clip(0, 3)

certifications = np.random.poisson(3, N).clip(0, 10)

aptitude_score = np.round(
    np.random.normal(70, 15, N).clip(30, 100),
    1
)

communication_score = np.round(
    np.random.normal(70, 14, N).clip(30, 100),
    1
)

technical_score = np.round(
    np.random.normal(68, 16, N).clip(30, 100),
    1
)


# ---------------------------------------------------------
# 3. CALCULATE A HIDDEN PLACEMENT SCORE
# ---------------------------------------------------------
# This is used only to generate realistic synthetic outcomes.

placement_score = (
    0.30 * cgpa
    + 0.12 * (dsa_problems / 50)
    + 0.12 * projects
    + 0.12 * internships
    + 0.05 * certifications
    + 0.09 * (aptitude_score / 10)
    + 0.08 * (communication_score / 10)
    + 0.12 * (technical_score / 10)
)

# Add randomness
placement_score += np.random.normal(0, 0.8, N)


# ---------------------------------------------------------
# 4. CONVERT SCORE INTO PLACEMENT PROBABILITY
# ---------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


probability = sigmoid(
    (placement_score - placement_score.mean()) / 1.5
)

placed = np.random.binomial(1, probability)

placement_status = np.where(
    placed == 1,
    "Placed",
    "Not Placed"
)


# ---------------------------------------------------------
# 5. GENERATE PACKAGE
# ---------------------------------------------------------

package_lpa = np.where(
    placed == 1,

    (
        3
        + (cgpa - 5) * 1.2
        + projects * 0.8
        + internships * 1.2
        + dsa_problems * 0.015
        + technical_score * 0.025
        + np.random.normal(0, 1.5, N)
    ),

    0
)

package_lpa = np.round(
    package_lpa.clip(2.5, 25),
    2
)


# ---------------------------------------------------------
# 6. CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame({
    "student_id": student_id,
    "cgpa": cgpa,
    "dsa_problems": dsa_problems,
    "projects": projects,
    "internships": internships,
    "certifications": certifications,
    "aptitude_score": aptitude_score,
    "communication_score": communication_score,
    "technical_score": technical_score,
    "placement_status": placement_status,
    "package_lpa": package_lpa
})


# ---------------------------------------------------------
# 7. SAVE DATASET
# ---------------------------------------------------------

df.to_csv(
    "data/placement_data.csv",
    index=False
)


# ---------------------------------------------------------
# 8. DISPLAY RESULTS
# ---------------------------------------------------------

print("=" * 50)
print("CAMPUS PLACEMENT DATASET CREATED")
print("=" * 50)

print(f"\nNumber of students: {len(df)}")

print("\nFirst 5 students:")
print(df.head())

print("\nPlacement distribution:")
print(df["placement_status"].value_counts())

print("\nPlacement percentage:")
print(
    df["placement_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nAverage CGPA:")
print(round(df["cgpa"].mean(), 2))

print("\nAverage package of placed students:")
print(
    round(
        df.loc[
            df["placement_status"] == "Placed",
            "package_lpa"
        ].mean(),
        2
    ),
    "LPA"
)

print("\nDataset saved to:")
print("data/placement_data.csv")

print("=" * 50)