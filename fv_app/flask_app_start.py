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

# home page
@app.route('/', methods=['GET'])
def home(): 
    return render_template('home.html')

# create a input box for text
@app.route('/', methods=['GET'])
def get_text():
    return render_template('submit.html')

# nutrition facts page
@app.route('/nutrition_facts', methods=['POST'])
def predict_nut_facts(X_train, X_test, y_train, y_test, grayscale, edge):
    """Render a page containing the nutrition facts for the chosen image or name"""
    data = str(request.form['nutri_facts'])
    pred = str(model.predict([data])[0])
    fru_veg_class.naive_bayes(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    return render_template('nutrition_facts.html', article=data, predicted=pred)

# contact information page
@app.route('/contact', methods=['GET'])
def contact_info():
    """Render a page containing the contact information."""
    return render_template('contact_info.html')

if __name__ == '__main__':
    data = 'data/nutri_facts_name.csv'
    get_dat_data = get_nutri_facts(data)
    X = []
    y = []
    grayscale = False
    edge = False
    all_fru_veg = os.listdir('data/fruits_vegetables')
    fru_veg_class = FruitsVeggiesNB(X, y, all_fru_veg)
    X, y, all_fru_veg = fru_veg_class.get_X_y_fv(X, y, all_fru_veg, grayscale=grayscale, edge=edge)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    filename = 'fv_app/fv_nb_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    model = loaded_model.score(X_test, y_test)

    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)

