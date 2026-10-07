import pandas as pd

# ---------------------------------------------------------
# 1. LOAD THE DATASET
# ---------------------------------------------------------

df = pd.read_csv("data/placement_data.csv")


# ---------------------------------------------------------
# 2. BASIC INFORMATION
# ---------------------------------------------------------

print("=" * 60)
print("CAMPUS PLACEMENT DATA ANALYSIS")
print("=" * 60)

print("\nTotal number of students:")
print(len(df))


# ---------------------------------------------------------
# 3. PLACEMENT DISTRIBUTION
# ---------------------------------------------------------

print("\nPlacement Distribution:")
print(df["placement_status"].value_counts())


# ---------------------------------------------------------
# 4. PLACEMENT RATE
# ---------------------------------------------------------

placement_rate = (
    df["placement_status"]
    .eq("Placed")
    .mean()
    * 100
)

print("\nPlacement Rate:")
print(round(placement_rate, 2), "%")


# ---------------------------------------------------------
# 5. AVERAGE VALUES
# ---------------------------------------------------------

print("\nAverage CGPA:")
print(round(df["cgpa"].mean(), 2))

print("\nAverage DSA Problems:")
print(round(df["dsa_problems"].mean(), 2))

print("\nAverage Projects:")
print(round(df["projects"].mean(), 2))

print("\nAverage Internships:")
print(round(df["internships"].mean(), 2))


# ---------------------------------------------------------
# 6. AVERAGE PACKAGE
# ---------------------------------------------------------

placed_students = df[
    df["placement_status"] == "Placed"
]

average_package = placed_students["package_lpa"].mean()

print("\nAverage Package of Placed Students:")
print(round(average_package, 2), "LPA")


# ---------------------------------------------------------
# 7. COMPARE PLACED VS NOT PLACED
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PLACED VS NOT PLACED")
print("=" * 60)

comparison = df.groupby(
    "placement_status"
)[
    [
        "cgpa",
        "dsa_problems",
        "projects",
        "internships",
        "certifications",
        "aptitude_score",
        "communication_score",
        "technical_score"
    ]
].mean()

print(comparison.round(2))


# ---------------------------------------------------------
# 8. DATASET INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset shape:")
print(df.shape)