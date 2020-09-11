# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples

from sklearn.model_selection import train_test_split
from flask import Flask, render_template
import sys
sys.path.append('src')
from class_fruit_veggies_NB import FruitsVeggiesNB
import os


app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html')

# nutrition facts page
@app.route('/nutrition_facts', methods=['POST'])
def naive_bayes(X_train, X_test, y_train, y_test, grayscale, edge):
    """Render a page containing a text area input where the user can paste an
    description to be classified."""
    fru_veg_class.naive_bayes(X_train, X_test, y_train, y_test, grayscale=grayscale, edge=edge)
    return render_template('nutrition_facts.html')

# contact information page
@app.route('/contact', methods=['GET'])
def contact_info():
    """Render a page containing the nutrition facts."""
    return render_template('contact.html')

if __name__ == '__main__':
    X = []
    y = []
    grayscale = False
    edge = False
    all_fru_veg = os.listdir('data/fruits_vegetables')
    fru_veg_class = FruitsVeggiesNB(X, y, all_fru_veg)
    X, y, all_fru_veg = fru_veg_class.get_X_y_fv(X, y, all_fru_veg, grayscale=grayscale, edge=edge)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True, )

