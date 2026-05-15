import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CareerVista",
    page_icon="💼",
    layout="wide"
)

# ---------------- LOAD FILES ----------------

model = joblib.load("job_salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

df = pd.read_csv("india_job_market_2024_2026.csv")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #312e81, #4338ca);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* TITLES */

.main-title {
    font-size: 60px;
    font-weight: bold;
    color: #111827;
}

.sub-title {
    font-size: 22px;
    color: #374151;
}

/* GLASS CARD */

.glass-card {
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* FEATURE BOX */

.feature-box {
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    transition: 0.3s;
}

.feature-box:hover {
    transform: scale(1.03);
}

/* RESULT BOX */

.result-box {
    background: linear-gradient(135deg,#0f172a,#312e81);
    padding: 35px;
    border-radius: 25px;
    color: white;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
}

/* BUTTON */

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    color: white;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(135deg,#4338ca,#6d28d9);
}

/* SELECT BOX */

div[data-baseweb="select"] {
    background-color: white;
    border-radius: 10px;
}

/* NUMBER INPUT */

div[data-baseweb="input"] {
    background-color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("💼 CareerVista")

menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "ℹ About", "📈 Prediction", "📊 Results", "🚀 Career Guide"]
)

# ---------------- HOME PAGE ----------------

if menu == "🏠 Home":

    st.markdown(
        '<p class="main-title">CareerVista</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">Smart Career Analytics and Salary Prediction Platform</p>',
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1521737604893-d14cc237f11d",
        use_container_width=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
        <h2>📊 Analytics</h2>
        <p>Explore industry salary trends and smart insights.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
        <h2>🤖 Predictions</h2>
        <p>Predict salary packages using Machine Learning.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
        <h2>🚀 Career Growth</h2>
        <p>Understand opportunities and improve professionally.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="glass-card">

    <h2>✨ Why Choose CareerVista?</h2>

    <p>
    CareerVista provides an intelligent and modern platform for users to explore career analytics,
    salary predictions, and industry insights with interactive visualizations and a beautiful experience.
    </p>

    </div>
    """, unsafe_allow_html=True)

# ---------------- ABOUT PAGE ----------------

elif menu == "ℹ About":

    st.markdown(
        '<p class="main-title">About Us</p>',
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1497366754035-f200968a6e72",
        use_container_width=True
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        option = st.selectbox(
            "Choose Information",
            [
                "Platform Overview",
                "Our Features",
                "Technology",
                "User Experience"
            ]
        )

        if option == "Platform Overview":

            st.markdown("""
            <div class="glass-card">

            <h3>🌟 Platform Overview</h3>

            <p>
            CareerVista is a modern intelligent platform designed for users to explore career analytics,
            salary estimation, and professional insights with a smooth and interactive experience.
            </p>

            </div>
            """, unsafe_allow_html=True)

        elif option == "Our Features":

            st.markdown("""
            <div class="glass-card">

            <h3>🔥 Features</h3>

            <ul>
            <li>Interactive Dashboard</li>
            <li>Salary Prediction</li>
            <li>Career Guidance</li>
            <li>Modern User Interface</li>
            <li>Data Visualization</li>
            </ul>

            </div>
            """, unsafe_allow_html=True)

        elif option == "Technology":

            st.markdown("""
            <div class="glass-card">

            <h3>💻 Technology</h3>

            <p>
            The platform uses Machine Learning, Streamlit, Plotly,
            Python, and intelligent analytics to provide a modern experience.
            </p>

            </div>
            """, unsafe_allow_html=True)

        elif option == "User Experience":

            st.markdown("""
            <div class="glass-card">

            <h3>🎨 User Experience</h3>

            <p>
            Designed with modern UI concepts, smooth navigation,
            responsive layouts, and visually appealing components.
            </p>

            </div>
            """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="glass-card">

        <h3>📌 Highlights</h3>

        <p>✔ Smart Analytics</p>
        <p>✔ Interactive Visualizations</p>
        <p>✔ AI-Based Salary Prediction</p>
        <p>✔ User Friendly Interface</p>
        <p>✔ Fast Performance</p>

        </div>
        """, unsafe_allow_html=True)

# ---------------- PREDICTION PAGE ----------------

elif menu == "📈 Prediction":

    st.markdown(
        '<p class="main-title">Salary Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass-card">
    Fill in the details below to estimate the expected salary package.
    </div>
    """, unsafe_allow_html=True)

    input_data = {}

    col1, col2 = st.columns(2)

    i = 0

    for column in model_columns:

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if column in label_encoders:

                options = label_encoders[column].classes_

                selected = st.selectbox(
                    column.replace("_", " "),
                    options
                )

                encoded_value = label_encoders[column].transform([selected])[0]

                input_data[column] = encoded_value

            else:

                value = st.number_input(
                    column.replace("_", " "),
                    min_value=0.0,
                    value=1.0
                )

                input_data[column] = value

        i += 1

    if st.button("Predict Salary"):

        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div class="result-box">

        <h2>💰 Estimated Salary Package</h2>

        <h1>₹ {prediction:.2f} LPA</h1>

        <p>Based on the selected job profile and experience.</p>

        </div>
        """, unsafe_allow_html=True)

# ---------------- RESULTS PAGE ----------------

elif menu == "📊 Results":

    st.markdown(
        '<p class="main-title">Insights Dashboard</p>',
        unsafe_allow_html=True
    )

    chart_option = st.selectbox(
        "Choose Visualization",
        [
            "Top Job Roles",
            "Salary Distribution"
        ]
    )

    if chart_option == "Top Job Roles":

        top_jobs = df['Job_Title'].value_counts().head(10)

        fig1 = px.bar(
            x=top_jobs.values,
            y=top_jobs.index,
            orientation='h',
            title="Top Job Roles"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    elif chart_option == "Salary Distribution":

        fig2 = px.histogram(
            df,
            x='Salary_LPA',
            nbins=30,
            title='Salary Distribution'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ---------------- EXTRA FEATURE ----------------

elif menu == "🚀 Career Guide":

    st.markdown(
        '<p class="main-title">Career Guide</p>',
        unsafe_allow_html=True
    )

    choice = st.selectbox(
        "Choose Category",
        [
            "Technical Skills",
            "Interview Preparation",
            "Resume Building",
            "Communication Skills"
        ]
    )

    if choice == "Technical Skills":

        st.markdown("""
        <div class="glass-card">

        <h3>💻 Technical Skills</h3>

        <p>✔ Python</p>
        <p>✔ Machine Learning</p>
        <p>✔ SQL</p>
        <p>✔ Data Analysis</p>
        <p>✔ Web Development</p>

        </div>
        """, unsafe_allow_html=True)

    elif choice == "Interview Preparation":

        st.markdown("""
        <div class="glass-card">

        <h3>🎯 Interview Preparation</h3>

        <p>✔ Practice Aptitude Questions</p>
        <p>✔ Mock Interviews</p>
        <p>✔ HR Questions</p>
        <p>✔ Technical Rounds</p>

        </div>
        """, unsafe_allow_html=True)

    elif choice == "Resume Building":

        st.markdown("""
        <div class="glass-card">

        <h3>📄 Resume Building</h3>

        <p>✔ Keep resume clean</p>
        <p>✔ Highlight projects</p>
        <p>✔ Mention technical skills</p>
        <p>✔ Use proper formatting</p>

        </div>
        """, unsafe_allow_html=True)

    elif choice == "Communication Skills":

        st.markdown("""
        <div class="glass-card">

        <h3>🗣 Communication Skills</h3>

        <p>✔ Improve speaking confidence</p>
        <p>✔ Practice presentations</p>
        <p>✔ Build networking skills</p>
        <p>✔ Improve professional communication</p>

        </div>
        """, unsafe_allow_html=True)