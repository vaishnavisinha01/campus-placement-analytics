import pandas as pd
import matplotlib.pyplot as plt

# Load our dataset
df = pd.read_csv("data/placement_data.csv")

# Count placed and not placed students
placement_counts = df["placement_status"].value_counts()

# Create the bar chart
plt.figure(figsize=(7, 5))

plt.bar(
    placement_counts.index,
    placement_counts.values
)

plt.title("Placement Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.show()

# 2. CGPA DISTRIBUTION


plt.figure(figsize=(8, 5))

plt.hist(
    df["cgpa"],
    bins=15
)

plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.show()

# 3. PROJECTS VS PLACEMENT


project_placement = pd.crosstab(
    df["projects"],
    df["placement_status"]
)

project_placement.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Projects vs Placement")
plt.xlabel("Number of Projects")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.show()

# 4. PLACEMENT RATE BY NUMBER OF PROJECTS


placement_rate_by_projects = (
    df.groupby("projects")["placement_status"]
    .apply(lambda x: (x == "Placed").mean() * 100)
)

print("\nPlacement Rate by Number of Projects:")
print(placement_rate_by_projects.round(2))


# Create graph
plt.figure(figsize=(8, 5))

plt.plot(
    placement_rate_by_projects.index,
    placement_rate_by_projects.values,
    marker="o"
)

plt.title("Placement Rate vs Number of Projects")
plt.xlabel("Number of Projects")
plt.ylabel("Placement Rate (%)")

plt.xticks(placement_rate_by_projects.index)

plt.grid(True)

plt.tight_layout()

plt.show()