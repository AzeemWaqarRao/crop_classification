import streamlit as st 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


model = load_model("vgg.h5")
crop = {0: 'Cherry',
 1: 'Coffee-plant',
 2: 'Cucumber',
 3: 'Fox_nut(Makhana)',
 4: 'Lemon',
 5: 'Olive-tree',
 6: 'Pearl_millet(bajra)',
 7: 'Tobacco-plant',
 8: 'almond',
 9: 'banana',
 10: 'cardamom',
 11: 'chilli',
 12: 'clove',
 13: 'coconut',
 14: 'cotton',
 15: 'gram',
 16: 'jowar',
 17: 'jute',
 18: 'maize',
 19: 'mustard-oil',
 20: 'papaya',
 21: 'pineapple',
 22: 'rice',
 23: 'soyabean',
 24: 'sugarcane',
 25: 'sunflower',
 26: 'tea',
 27: 'tomato',
 28: 'vigna-radiati(Mung)',
 29: 'wheat'}


def preprocess_image(image):
    image = load_img(image,target_size = [196,196,3])
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    return image


def main():
    st.title("Crop Classification")
    image = st.file_uploader("Upload Image")
    if image:
        img = preprocess_image(image)
        result = model.predict(img)
        result = np.argmax(result)
        st.success("The Crop is : "+ crop[result])
        st.image(image)


if __name__ == "__main__":
    main()