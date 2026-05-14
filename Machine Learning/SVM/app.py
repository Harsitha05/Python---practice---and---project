import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

page_style = """
<style>

[data-testid="stAppViewContainer"]{
background-image: linear-gradient(
rgba(0,0,0,0.45),
rgba(0,0,0,0.45)),
url("https://images.unsplash.com/photo-1468327768560-75b778cbb551?q=80&w=1974&auto=format&fit=crop");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

h1{
text-align:center;
color:white;
font-size:55px;
font-weight:bold;
margin-top:20px;
}

p{
text-align:center;
color:#F3F4F6;
font-size:20px;
font-weight:500;
}

h3{
color:white !important;
font-size:35px !important;
margin-top:30px;
}

.stSlider label{
color:white !important;
font-size:18px !important;
font-weight:bold;
}

.stSlider > div[data-baseweb="slider"] > div{
background: linear-gradient(to right, #EC4899, #8B5CF6);
height:8px;
border-radius:10px;
}

.stButton>button{
width:100%;
background: linear-gradient(to right, #EC4899, #8B5CF6);
color:white;
font-size:20px;
font-weight:bold;
padding:12px;
border:none;
border-radius:15px;
transition:0.3s;
margin-top:20px;
}

.stButton>button:hover{
transform:scale(1.05);
background: linear-gradient(to right, #8B5CF6, #EC4899);
}

.result-box{
background: rgba(255,255,255,0.2);
backdrop-filter: blur(10px);
padding:15px;
border-radius:15px;
text-align:center;
font-size:25px;
font-weight:bold;
color:white;
margin-top:20px;
box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

</style>
"""

st.markdown(page_style, unsafe_allow_html=True)

st.title("🌸 Iris Flower Prediction")

st.write("Predict the species of Iris flower using Machine Learning")

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

st.subheader("Flower Measurements")

sepal_length = st.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.5
)

sepal_width = st.slider(
    "Sepal Width",
    2.0,
    5.0,
    3.2
)

petal_length = st.slider(
    "Petal Length",
    1.0,
    7.0,
    4.5
)

petal_width = st.slider(
    "Petal Width",
    0.1,
    3.0,
    1.2
)

input_data = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
])

input_data = scaler.transform(input_data)

prediction = model.predict(input_data)

species = [
    "🌼 Setosa",
    "🌷 Versicolor",
    "🌺 Virginica"
]

if st.button("Predict Flower"):
    st.markdown(
        f'<div class="result-box">Prediction : {species[prediction[0]]}</div>',
        unsafe_allow_html=True
    )