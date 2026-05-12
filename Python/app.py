import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import os

st.set_page_config(page_title="Titanic Sea Voyage", page_icon="⚓", layout="centered")

def apply_theme(status=None):
    if status == "survived":
        overlay = "rgba(40, 167, 69, 0.3)" 
    elif status == "died":
        overlay = "rgba(220, 53, 69, 0.3)"   
    else:
        overlay = "rgba(0, 50, 100, 0.5)"   

    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient({overlay}, {overlay}), 
                        url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1920&q=80");
            background-size: cover;
            background-attachment: fixed;
        }}
        h1, h2, h3, p, label, .stMarkdown {{
            color: white !important;
            text-shadow: 2px 2px 4px #000000;
            font-weight: bold !important;
        }}
        .stNumberInput, .stSelectbox, .stSlider, .stRadio {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 10px;
            backdrop-filter: blur(5px);
        }}
        </style>
        """, unsafe_allow_html=True)

if not os.path.exists("titanic_model.pkl"):
    try:
        df = pd.read_csv(r"C:\Users\ACER\Downloads\titanic.csv")
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Embarked'] = df['Embarked'].fillna(0)
        
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        X = df[features]
        y = df['Survived']
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        joblib.dump(model, "titanic_model.pkl")
    except Exception as e:
        st.error(f"Initialization Error: {e}")

try:
    model = joblib.load("titanic_model.pkl")
except:
    st.error("Model file tracking failed.")

apply_theme() 

st.markdown("<h1 style='text-align: center;'>🚢 TITANIC SURVIVAL PREDICTOR ⚓</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter details below to see if the passenger survives the voyage.</p>", unsafe_allow_html=True)

st.markdown("###  Ticket & Cabin Info")
col1, col2 = st.columns(2)
with col1:
    pclass = st.radio("Select Passenger Class ", [1, 2, 3], help="1=First Class, 2=Second, 3=Third")
    fare = st.number_input("Ticket Fare ($) ", 0.0, 512.0, 32.0)

with col2:
    embarked = st.selectbox("Departure Port ", ["S", "C", "Q"], 
                            format_func=lambda x: "Southampton" if x=="S" else "Cherbourg" if x=="C" else "Queenstown")
    sex = st.selectbox("Gender ", ["male", "female"])

st.markdown("### Passenger Personal Details")
col3, col4 = st.columns(2)
with col3:
    age = st.slider("Passenger Age ", 0, 100, 25)

with col4:
    sibsp = st.number_input("Siblings/Spouses Aboard ", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard ", 0, 6, 0)

sex_val = 0 if sex == "male" else 1
emb_map = {"S": 0, "C": 1, "Q": 2}
emb_val = emb_map[embarked]

st.markdown("---")

if st.button("🔮 PREDICT MY FATE", use_container_width=True):
    input_data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, emb_val]], 
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        apply_theme("survived")
        st.balloons()
        st.success(f"## 🎉 SURVIVED!")
        st.markdown(f"### Probability of Survival: {prob:.1f}%")
    else:
        apply_theme("died")
        st.error(f"##  DID NOT SURVIVE")
        st.markdown(f"### Probability of Survival: {prob:.1f}%")

    st.info(f"Summary: A {age} year old {sex} in class {pclass} paying ${fare:.2f} had a {prob:.1f}% chance of survival.")