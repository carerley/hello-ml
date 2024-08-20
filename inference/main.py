from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import tensorflow_hub as hub

import kagglehub

# Download latest version
import tensorflow as tf
import numpy as np
from PIL import Image

def main():


    # Assume the model has been trained and saved as 'sentiment_model.joblib'
    # and the vectorizer has been saved as 'vectorizer.joblib'
    path = kagglehub.model_download("alfathterry/distilbert-amazon-review-positive-negative/tensorFlow2/v1")
    print("Path to model files:", path)

    return
    # Load the trained model and vectorizer
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')

    # New, unseen data for inference
    new_texts = ["I love this product!", "This is the worst purchase I've ever made."]

    # Transform the new texts using the same vectorizer
    new_texts_transformed = vectorizer.transform(new_texts)

    # Perform inference to get predictions
    predictions = model.predict(new_texts_transformed)

    # Output predictions
    for text, prediction in zip(new_texts, predictions):
        print(f"Text: {text} => Sentiment: {prediction}")


def main2():


    # Load the pre-trained model from a .h5 file
    model = tf.keras.models.load_model('tf_model.h5')



    # Define a function to preprocess the image
    def preprocess_image(image_path):
        image = Image.open(image_path).resize((224, 224))  # Resize to match the input size expected by the model
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    # Perform inference on a new image
    image_path = 'path/to/your/image.jpg'
    image = preprocess_image(image_path)
    predictions = model.predict(image)

    # Print the predictions
    print(predictions)


if __name__ == '__main__':
    main2()