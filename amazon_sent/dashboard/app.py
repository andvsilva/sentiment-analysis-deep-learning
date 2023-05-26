import streamlit as st
import joblib
import numpy as np
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.models import load_model

# Load the trained model
model = keras.models.load_model('modelkeras.h5')

def main():
    # Set the app title
    st.title('Deep Learning Model Deployment')
    
    # Create an input text field
    text_input = [st.text_input("Enter text", "Type here...")]
    ## Convert a collection of text documents to a matrix of token counts.
    cv = CountVectorizer()

    # feature
    validation = cv.fit_transform(text_input)
    
    # Make predictions on the input text
    if st.button("Predict"):
        prediction = model.predict(model, validation)  # Replace with your prediction code
        st.write("Prediction:", prediction)
    
if __name__ == '__main__':
    main()