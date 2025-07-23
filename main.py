import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pretrained model with relu activation
model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    """Decode a review from integers back to words"""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """Preprocess text input for the model"""
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    """Predict sentiment of a review"""
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review', placeholder="Enter your movie review here...")

if st.button('Classify'):
    if user_input.strip():  # Check if input is not empty
        try:
            sentiment, score = predict_sentiment(user_input)
            
            st.write(f'**Sentiment:** {sentiment}')
            st.write(f'**Prediction Score:** {score:.4f}')
            
            # Add some visual feedback
            if sentiment == 'Positive':
                st.success(f'This review is classified as {sentiment}! 😊')
            else:
                st.error(f'This review is classified as {sentiment}! 😞')
                
            # Show confidence level
            confidence = max(score, 1 - score)
            st.write(f'**Confidence:** {confidence:.2%}')
            
        except Exception as e:
            st.error(f'An error occurred: {str(e)}')
    else:
        st.warning('Please enter a movie review to classify.')
else:
    st.info('Please enter a movie review and click "Classify" to get started.')

# Add some example reviews for testing
st.subheader('Try these example reviews:')
col1, col2 = st.columns(2)

with col1:
    if st.button('Example: Positive Review'):
        st.session_state.example_text = "This movie was fantastic! The acting was great and the plot was thrilling. I loved every minute of it."

with col2:
    if st.button('Example: Negative Review'):
        st.session_state.example_text = "This movie was terrible. The acting was poor and the plot was boring. I wasted my time watching it."

# Update text area with example if button was clicked
if hasattr(st.session_state, 'example_text'):
    user_input = st.text_area('Movie Review', value=st.session_state.example_text, placeholder="Enter your movie review here...")
    # Clear the example text after use
    if st.session_state.example_text:
        st.session_state.example_text = ""