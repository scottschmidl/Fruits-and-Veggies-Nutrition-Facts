# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request,  redirect, url_for
import os
from werkzeug.utils import secure_filename
import pandas as pd
import sys
sys.path.append('src')
from class_fruit_veggies_NB import FruitsVeggiesNB
from nutri_facts_df import get_nutri_facts
import pickle

app = Flask(__name__)

# navie bayes model to run through flask app
filename = 'fv_app/fv_nb_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# home page
@app.route('/', methods=['GET'])
def home(): 
    return render_template('home.html')

# create a input box for images
@app.route('/submit', methods=['GET'])
def get_image():
    return render_template('submit.html')

# nutrition facts page
@app.route('/nutrition_facts', methods=['POST'])
def predict_nut_facts():
    """receive the image to be classified from input form and use model to classify.
    Once classified return nutrition facts about selected image"""
    data_ = request.form['nutr_facts']
    pred = loaded_model.predict([data_])[0]
    # contains dataframe with nutrition facts
    # df = 'data/nutri_facts_name.csv'
    # get_dat_data = get_nutri_facts(df)
    return render_template('nutrition_facts.html', image=data_, predicted=pred)

# contact information page
@app.route('/contact', methods=['GET'])
def contact_info():
    """Render a page containing the contact information."""
    return render_template('contact_info.html')

if __name__ == '__main__':
    # run the flask app
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)