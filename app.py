import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# PAGE CONFIG

st.set_page_config(
    page_title="Campus Placement Analytics",
    page_icon="🎓",
    layout="wide"
)


# LOAD DATA AND MODEL

df = pd.read_csv("data/placement_data.csv")
model = joblib.load("placement_model.pkl")

features = [
    "cgpa",
    "dsa_problems",
    "projects",
    "internships",
    "certifications",
    "aptitude_score",
    "communication_score",
    "technical_score"
]


# TITLE

st.title("🎓 Campus Placement Analytics & Prediction")

st.write(
    "An interactive dashboard for analyzing student placement data "
    "and predicting placement outcomes using Machine Learning."
)

st.divider()


# DATASET OVERVIEW

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", len(df))

with col2:
    placement_rate = (
        df["placement_status"] == "Placed"
    ).mean() * 100

    st.metric(
        "Placement Rate",
        f"{placement_rate:.1f}%"
    )

with col3:
    st.metric(
        "Average CGPA",
        f"{df['cgpa'].mean():.2f}"
    )

with col4:
    avg_package = df.loc[
        df["placement_status"] == "Placed",
        "package_lpa"
    ].mean()

    st.metric(
        "Avg Package",
        f"{avg_package:.2f} LPA"
    )

st.divider()


# VISUAL ANALYSIS

st.subheader("📈 Placement Analysis")

col1, col2 = st.columns(2)

# Placement Distribution
with col1:
    st.write("### Placement Distribution")

    placement_counts = df["placement_status"].value_counts()

    fig, ax = plt.subplots()

    placement_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Number of Students")
    ax.set_title("Placed vs Not Placed")

    st.pyplot(fig)

# Projects vs Placement
with col2:
    st.write("### Projects vs Placement")

    project_rate = (
        df.groupby("projects")["placement_status"]
        .apply(lambda x: (x == "Placed").mean() * 100)
    )

    fig, ax = plt.subplots()

    project_rate.plot(
        kind="line",
        marker="o",
        ax=ax
    )

    ax.set_xlabel("Number of Projects")
    ax.set_ylabel("Placement Rate (%)")
    ax.set_title("Placement Rate vs Projects")

    st.pyplot(fig)

st.divider()


# PLACEMENT PREDICTION

st.subheader("🤖 Placement Prediction")

st.write(
    "Enter a student's profile below to predict their placement outcome."
)

col1, col2, col3, col4 = st.columns(4)

# Column 1
with col1:

    cgpa = st.slider(
        "CGPA",
        5.0,
        10.0,
        7.5,
        0.1
    )

    dsa = st.number_input(
        "DSA Problems Solved",
        min_value=0,
        max_value=500,
        value=100
    )

# Column 2
with col2:

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=10,
        value=2
    )

    internships = st.number_input(
        "Internships",
        min_value=0,
        max_value=5,
        value=1
    )

# Column 3
with col3:

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=15,
        value=2
    )

    aptitude = st.slider(
        "Aptitude Score",
        0,
        100,
        70
    )

# Column 4
with col4:

    communication = st.slider(
        "Communication Score",
        0,
        100,
        70
    )

    technical = st.slider(
        "Technical Score",
        0,
        100,
        70
    )


# PREDICTION BUTTON

if st.button(
    "🔮 Predict Placement",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            cgpa,
            dsa,
            projects,
            internships,
            certifications,
            aptitude,
            communication,
            technical
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == "Placed":

        st.success("🎉 Prediction: PLACED")

    else:

        st.error("❌ Prediction: NOT PLACED")

    # Find probability of "Placed"
    placed_index = list(model.classes_).index("Placed")

    placed_probability = probability[placed_index]

    st.write(
        f"### Placement Probability: "
        f"{placed_probability * 100:.2f}%"
    )

    st.progress(
        float(placed_probability)
    )


# SAMPLE DATA

st.divider()

st.subheader("🔍 Sample Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)