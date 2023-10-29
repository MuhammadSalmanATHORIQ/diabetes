import pickle
import streamlit as st
# Load the diabetes model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

st.title('Data Mining Prediksi Diabetes')

# Create input fields using st.columns
col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.text_input('Input Nilai Pregnancies')
    BloodPressure = st.text_input('Input Nilai BloodPressure')
    Insulin = st.text_input('Input Nilai Insulin')
    DiabetesPedigreeFunction = st.text_input('Input Nilai DiabetesPedigreeFunction')
with col2:
    Glucose = st.text_input('Input Nilai Glucose')
    SkinThickness = st.text_input('Input Nilai SkinThickness')
    BMI = st.text_input('Input Nilai BMI')
    Age = st.text_input('Input Nilai Age')

predict = ''

if st.button('Cek Analisis Diabetes'):
    # Convert input values to numbers
    input_data = [[
        float(Pregnancies), float(Glucose), float(BloodPressure),
        float(SkinThickness), float(Insulin), float(BMI),
        float(DiabetesPedigreeFunction), float(Age)
    ]]
    predict = diabetes_model.predict(input_data)
    st.write('Estimasi diabetes:', predict)
