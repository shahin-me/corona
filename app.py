import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
classifier = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prob = classifier.predict_proba(final_features)
    prob = prob[:, 1]
    prob = prob * 100

    #prediction = classifier.predict(final_features)

    output = round(prob[0], 2)

    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    return render_template('index.html', prediction_text='Yout corona virus is {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)