import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import streamlit as st

# ignore warnings
warnings.filterwarnings('ignore')

# Loading Data files divided into test and train
X_test = pickle.load(open('data/X_test', 'rb'))
X_train = pickle.load(open('data/X_train', 'rb'))

# standardizing the data for faster and accurate result
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Loading the Trained model
model = pickle.load(open('TrainedModel.sav', 'rb'))
def calories_prediction(input_data):
    data_array = np.asarray(input_data)
    data_reshaped = data_array.reshape(1, -1)
    data_transformed = scaler.transform(data_reshaped)
    prediction = model.predict(data_transformed)
    return prediction


st.title('Calories Burnt Prediction using AI')
st.sidebar.image('usrImage/hmrlogo.png')
st.sidebar.write('# Summer Training Project')
st.sidebar.write('# Shivam Singh\n # 07313302720\n # CSE-5A\n')
gender = st.radio(
        "What is your gender",
        ('Male', 'Female')
)
if gender == 'Male':
    gender = 0
else:
    gender = 1
age = st.text_input('Write your age')
height = st.text_input('Write your height in cm')
weight = st.text_input('Write your weight in Kg')
duration = st.text_input('Duration of exercise')
heart_rate = st.text_input('Average heart rate')
body_temp = st.text_input('Average body temp in celsius')


calculated_calories = ''

# creating the result button
if st.button('Calculate the calories burnt during exercise'):
    calculated_calories = calories_prediction([gender, age, height, weight, duration, heart_rate, body_temp])

st.success(calculated_calories)
