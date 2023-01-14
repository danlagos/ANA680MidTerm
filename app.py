from flask import Flask, render_template, request
import numpy as np
#import pickle
import joblib

app = Flask(__name__)

filename = 'NB.pkl'

model = joblib.load(filename)

@app.route('/')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    reading_score = request.form['reading_score']
    writing_score = request.form['writing_score']
    math_score = request.form['math_score']
          
    pred = model.predict(np.array([[reading_score, writing_score, math_score]], dtype=float))
    print(pred)
    return render_template('index.html', predict=str(pred))

if __name__ == '__main__':
    app.run