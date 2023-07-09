import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from flask import Flask, request, jsonify
import json

data = pd.read_csv('startups.csv')
data['State'] = data['State'].map({'New York': 1, 'California':2,'Florida':3})

feature = data.drop('Profit',axis = 1)
target = data.Profit

# le = LabelEncoder()
# x = le.fit_transform(feature)
# y = le.fit_transform(target)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size = 0.20)


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)

# sv = SVC(kernel='linear').fit(x_train,y_train)

prediction = model.predict(x_test)

app = Flask(__name__)
model = pickle.load(open("StartUp.pkl",'rb'))






def predict_charges(RnD, Administration, Marketing, State):
  inputan = {'R&D Spend':[RnD],'Administration':[Administration],'Marketing Spend':[Marketing],'State':[State]}
  akhiran = pd.DataFrame(inputan)
  return float(model.predict(Unilever))

# def load_save_artifacts():
#     print("Loading save model.. start")

#     with open("/model/columns.json",'r') as f:
#         __data_columns = json.load(f)['data_columns']
#         __regions = __data_columns[5:]
#     global __model
#     with open("/model/MC.pickle",'rb') as f:
#         __model = pickle.load(f)
#         print("loading saved model...done")


# if __name__ == '__main__':
#     load_save_artifacts()
#     print(get_region())
#     print(get_estimatied_price('southwest',26,1,27,3,1))
#     print(get_estimatied_price('southwest', 26, 1, 27, 3, 0))