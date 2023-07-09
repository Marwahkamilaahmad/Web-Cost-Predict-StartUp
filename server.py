import numpy as np
import util
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)

model = pickle.load(open('StartUp.sav','rb'))
# encoding='bytes'

@app.route('/')
def index():
    return render_template('app.html', **locals())

@app.route('/predict',methods=['POST', 'GET'])
def predict():

    RnD = float(request.form['RnD'])
    Administration = float(request.form['Administration'])
    Marketing = float(request.form['Marketing'])
    State = int(request.form['State'])
    arr = np.array([[RnD,Administration,Marketing,State]])

    results = model.predict(arr)
    return render_template('app.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)


