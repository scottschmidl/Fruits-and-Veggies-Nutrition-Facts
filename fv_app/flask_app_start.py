# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from flask import Flask, render_template, request,  redirect, url_for
from skimage.transform import resize
from skimage import io
import pandas as pd
import numpy as np
import tensorflow
import pickle
import sys
import os
sys.path.append('src')
from nutri_facts_df import get_nutri_facts

app = Flask(__name__)

## navie bayes model to run through flask app
filename_nb = 'fv_app/fv_nb_model.sav'
loaded_model_nb = pickle.load(open(filename_nb, 'rb'))

## convolutional neural network model to run through flask app
filename_cnn = 'fv_app/fv_cnn_model.sav'
load_model_cnn = tensorflow.keras.models.load_model('fv_app/fv_cnn_model.sav')

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
    """Receive the image to be classified from input form. Use model to classify
        Once classified return nutrition facts about selected image"""
    for uploaded_file in request.files.getlist('image'):
        if uploaded_file.filename != '':
            uploaded_file.save('fv_app/static/uploads/what_fruit_veggie_am_I.png')
    img = io.imread('fv_app/static/uploads/what_fruit_veggie_am_I.png')
    size = resize(img, (32, 32))
    ravel = size.ravel()
    ########### MNB Loaded Goods ######################
    pred_mnb = loaded_model_nb.predict([ravel])[0]
    print('pred_mnb*******************', pred_mnb)
    ########### End OF MNB Loaded Goods ###############
    ########### CNN Loaded Goods ######################
    vals = ['Pear', 'Tomato']
    cnn_size = resize(img, (32, 32, 32, 3))
    pred_cnn = load_model_cnn.predict([cnn_size])[0]
    print('pred_cnn*******************', pred_cnn)
    new_pred = np.argmax(pred_cnn)
    print('new_pred**************', new_pred)
    # new_pred_cnn = vals[new_pred] ## not working yet, getting a shape error
    final = pd.DataFrame({'name' : np.array(vals),'probability' :pred_cnn[0]})
    final, new_pred_cnn  =final.sort_values(by = 'probability',ascending=False), vals[new_pred]
    print('new_pred_cnn*****************', new_pred_cnn)
    print('final', final)
    ########### End Of CNN Loaded Goods ###############
    ########### Loaded Data Frame #####################
    nutri_facts_filename = 'data/nutri_facts_name.csv'
    df = get_nutri_facts(nutri_facts_filename)
    nf_mnb = df[df['Fruits_Vegetables_Name'] == pred_mnb]['Nutrition_Facts']
    print(nf_mnb)
    mnb_nf = nf_mnb.iloc[0]
    nf_cnn = df[df['Fruits_Vegetables_Name'] == new_pred_cnn]['Nutrition_Facts']
    cnn_nf = nf_cnn.iloc[0]
    print(cnn_nf)
    ########### End of Loaded Data Frame ##############
    return render_template('nutrition_facts.html', predicted=pred_mnb, fv_nf_mnb=mnb_nf)

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
    app.run(host='0.0.0.0', port=8150, threaded=True, debug=True)
