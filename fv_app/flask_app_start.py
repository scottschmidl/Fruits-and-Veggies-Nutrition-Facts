# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from flask import Flask, render_template, request,  redirect, url_for
import os
import pandas as pd
import sys
from skimage import io
from skimage.transform import resize
sys.path.append('src')
from class_fruit_veggies_NB import FruitsVeggiesNB
from nutri_facts_df import get_nutri_facts
import pickle
# import tensorflow

app = Flask(__name__)

## navie bayes model to run through flask app
filename_nb = 'fv_app/fv_nb_model.sav'
loaded_model_nb = pickle.load(open(filename_nb, 'rb'))

## convolutional neural network model to run through flask app
# filename_cnn = 'fv_app/fv_cnn_model.sav'
# load_model_cnn = tensorflow.keras.models.load_model('fv_app/fv_cnn_model.sav')

## home page
@app.route('/', methods=['GET'])
def home():
    """render the home screen"""
    return render_template('home.html')

## create an input box for images
@app.route('/submit', methods=['GET'])
def get_image():
    """render submit template with buttons for fruit and vegetable model"""
    return render_template('submit.html')

## nutrition facts page
@app.route('/nutrition_facts', methods=['POST'])
def predict_nut_facts():
    """Receive the image to be classified from input form. Use model to classify. 
        Once classified return nutrition facts about selected image"""        
    for uploaded_file in request.files.getlist('image'):        
        if uploaded_file.filename != '':
            uploaded_file.save('fv_app/static/uploads/what_fruit_veggie_am_I.png')
    img = io.imread('fv_app/static/uploads/what_fruit_veggie_am_I.png')        
    size = resize(img, (32, 32))
    ravel = size.ravel()
    pred = loaded_model_nb.predict([ravel])[0]
    # pred_cnn = load_model_cnn.evaluate([ravel])[0]
    ## contains dataframe with nutrition facts
    nutri_facts_filename = 'data/nutri_facts_name.csv'
    df = get_nutri_facts(nutri_facts_filename)
    nf = df[df['Fruits_Vegetables_Name'] == pred]['Nutrition_Facts']
    the_nutrition_fact = nf.iloc[0]
    return render_template('nutrition_facts.html', predicted=pred, fv=the_nutrition_fact)

## pear nutrition facts page
@app.route('/pear', methods=['POST'])
def pear_facts():
    '''returns nutrition facts for pears'''
    nutri_facts_filename = 'data/nutri_facts_name.csv'
    df = get_nutri_facts(nutri_facts_filename)
    pear_nf = df[df['Fruits_Vegetables_Name'] == "Pear"]['Nutrition_Facts']
    the_pear = pear_nf.iloc[0]
    return render_template('pear.html', peary_good=the_pear)

## tomato nutrition facts page
@app.route('/tomato', methods=['POST'])
def tomato_facts():
    '''returns nutrition facts for tomatoes'''
    nutri_facts_filename = 'data/nutri_facts_name.csv'
    df = get_nutri_facts(nutri_facts_filename)
    tomato_nf = df[df['Fruits_Vegetables_Name'] == "Tomato"]['Nutrition_Facts']
    the_tomato = tomato_nf.iloc[0]
    return render_template('tomato.html', tomato_goodness=the_tomato)

## full nutrition facts page
@app.route('/word_cloud', methods=['POST'])
def word_cloud_facts():
    '''returns nutrition facts for each fruit on file'''
    nutri_facts_filename = 'data/nutri_facts_name.csv'
    full_nf = get_nutri_facts(nutri_facts_filename)
    full_names = full_nf['Fruits_Vegetables_Name'].tolist()
    full_facts = full_nf['Nutrition_Facts'].tolist()
    name_fact_dict = dict(zip(full_names, full_facts))    
    return render_template('word_cloud.html', full_names_facts=name_fact_dict)

## contact information page
@app.route('/contact', methods=['GET'])
def contact_info():
    """Render a page containing my contact information."""
    return render_template('contact_info.html')

if __name__ == '__main__':
    ## run the flask app
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)