# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:40:07 2024

@author: lenovo
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model

loaded_model = pickle.load(open('C:/Users\lenovo\Downloads/deploy ML-Model/trained_model.sav', 'rb'))


# creating a function for predicting

def diabetes_prediction(input_data):

    # changing the input data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    print(prediction)

    if prediction[0] == 0:
      return 'Person is Non diabetic'

    else:
      return "Person is  diabetic"
  
def main():
    
    
    # Giving the title 
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user
    Pregnancies = st.text_input('Number Of Pregnancies')
    
    Glucose = st.text_input('Number Of Glucose')
    
    BloodPressure = st.text_input('Number Of BloodPressure')
    
    SkinThickness = st.text_input('Number Of SkinThickness')
    
    Insulin = st.text_input('Number Of Insulin')
    
    BMI = st.text_input('Number Of BMI')
    
    
    DiabetesPedigreeFunction = st.text_input('Number Of DiabetesPedigreeFunction')
    
    Age = st.text_input('Number Of Age')
    
    # Code of Prediction
    
    diagoniss = ''
    # creating a button for the Predition
    
    if st.button('Diabetes Test Result'):
        diagoniss = diabetes_prediction([Pregnancies, Glucose, BloodPressure,	SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,	Age	])
        
        
    st.success(diagoniss)
   

    

if __name__ == '__main__':
    main()
   

    
    


    


    

    

    