![4e539b07b5cf379069dc5e9c1d822a3](https://user-images.githubusercontent.com/97984680/181413098-4d8c6ab9-d7cf-4bb2-a090-787dc4738e1a.png)

This project aim to build an app to predict the medical cost in United States base on attributes: Age, Gender, Bmi, Reigon, The Number of Children, Smoke or not

![5796baf071a13330377e4ed711fe7fa](https://user-images.githubusercontent.com/97984680/181418807-937ae8f3-bb0a-4569-80ee-b2c22a913592.png)


# Dataset Introduction

Columns(7):
- age(Age of the customer)

- sex(Gender)

- bmi(Body Mass Index, an important health factor)

- children(number of children)

- smoker(whether the customer smokes or not)

- region(which region of the country the customer belongs to)

- charges(Target variable, the expenditure for the customer)

Source: https://www.kaggle.com/datasets/harshsingh2209/medical-insurance-payout


Link for this work in my kaggle: https://www.kaggle.com/code/eames07/medical-payout-prediction

# Exploratory Data Analysis

## Correlation Heatmap
```python
#Change categorical value into int
df['smoker']=Expenditure.smoker.map(dict(yes=1, no=0))
df['sex']=Expenditure.sex.map(dict(male=1, female=0))

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), vmin=-1, cmap="YlGnBu_r", annot=True)
```
![e6ae2418d47ff3b6277ac9d11b693e1](https://user-images.githubusercontent.com/97984680/181414041-6e3f5584-a25f-4ead-9f43-1316efe2b810.png)


## Distribution of each attributes
The number of children
```python
Children=Expenditure.children.value_counts().rename_axis('NumberofChildren').reset_index(name='count')
plt.figure(figsize=(10,10))
colors = ['#FFF8DC', '#9BCD9B', '#66CDAA', '#20B2AA',"#009ACD","#104E8B"]

plt.pie(Children["count"].tolist(), labels=Children["NumberofChildren"].tolist(), labeldistance=1.15, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, colors=colors,
       autopct='%1.1f%%',shadow=True);
plt.title("Number of Children")
```
![ef9bda2fcc4579b9d8f24c7fd9b30d2](https://user-images.githubusercontent.com/97984680/181414350-7928eed5-c0d7-4820-b7f4-dd6cc83eab14.png)


Region
```python
region=Expenditure.region.value_counts().rename_axis('region').reset_index(name='count')
plt.figure(figsize=(10,10))
colors = ['#FFF8DC', '#9BCD9B', '#66CDAA', '#20B2AA']

plt.pie(region["count"].tolist(), labels=region["region"].tolist(), labeldistance=1.15, wedgeprops = { 'linewidth' : 1, 'edgecolor' : 'white' }, colors=colors,
       autopct='%1.1f%%', shadow=True);
plt.title("Region Distribution")
```

![c718f2c805e37c276c192477a04fb70](https://user-images.githubusercontent.com/97984680/181414383-2d516766-225a-45c7-9317-26ca5b6f0692.png)

Age
```python
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
sns.distplot(Expenditure.age, kde = True, color='#0000FF', kde_kws={'color':'black'})
plt.show()
```
![a6de52a5a78c8118715a20a2c9c029b](https://user-images.githubusercontent.com/97984680/181414412-3b8fbaba-a8bb-4a3d-b7af-33cb5e9858d8.png)

BMI
```python
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
sns.distplot(Expenditure.bmi, kde = True, color='#0000FF', kde_kws={'color':'black'})
plt.show()
```
![109855ab825bec6f3c53bb4f913dc94](https://user-images.githubusercontent.com/97984680/181414427-ae93a44c-0500-494e-a9d4-8cbf97af59da.png)

Smoker
```python
plt.figure(figsize=(8,5))
sns.countplot(x='smoker', data = Expenditure, palette='GnBu', saturation=0.8)
```
![752225102a1fb5cd882f83eaacb3eaa](https://user-images.githubusercontent.com/97984680/181414451-2bc96c12-0e6c-444f-8086-9eaab66056ec.png)

Gender
```python
plt.figure(figsize=(8,5))
sns.countplot(x='sex', data = Expenditure, palette='GnBu', saturation=0.8)
```
![40c93c19902664c3a0d8d1b8c1e9ec0](https://user-images.githubusercontent.com/97984680/181414460-02952cb9-b0f2-4703-b0cb-fa3e51445975.png)


Medicial Cost
```python
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
sns.distplot(Expenditure.charges, kde = True, color='#0000FF', kde_kws={'color':'black'})
plt.show()
```
![f6604b087ab07919a4c053940353218](https://user-images.githubusercontent.com/97984680/181414480-27b34d06-66a9-429c-834f-ffc3496abb0e.png)

# Model Building
```python
#convert "region" in to dummies columns
dummies = pd.get_dummies(df.region)
df1 = pd.concat([df.drop('region',axis='columns'),dummies],axis='columns')
```

```python

#create linear regression model and test the prediction accuarcy
x=df1.drop(columns="charges")
y=df1.charges
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(x_train,y_train)
LR.score(x_test,y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), x, y, cv=cv)
```
![43e3dcaea27625c114d165cbe8acf43](https://user-images.githubusercontent.com/97984680/181414925-61727668-8ad1-401e-a581-9c3d0edf418d.png)

```python
#try to find the best model between decision, linear regression and lasso
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score'])

find_best_model_using_gridsearchcv(x,y)
```
![f68420c469cfa363f8b6a5b6ef5bbf8](https://user-images.githubusercontent.com/97984680/181415090-ff757373-dbac-4b13-a8ef-f6d4fa7d8594.png)

```python
#Build the prediction function
def predict_charges(region,age,sex,bmi,children,smoker):    
    reg_index  = np.where(x.columns==region)[0][0]

    a = np.zeros(len(x.columns))
    a[0] = age
    a[1] = sex
    a[2] = bmi
    a[3] = children
    a[4] = smoker
    if reg_index >= 0:
        a[reg_index] = 1

    return LR.predict([a])[0]
```

```python
#the medical cost prediction for person(age 23, female, bim 27, have 0 children, smoke) 
predict_charges('southwest',23, 0, 27, 0, 1)
```
![87e51160c0485ef6884f9a09f3b8c02](https://user-images.githubusercontent.com/97984680/181415247-d9c1bd1e-1f7f-4e6b-82c3-8331fe446af7.png)

```python
#the medical cost prediction for person(age 37, male, bim 35, have 2 children, do not smoke) 
predict_charges('southwest',37, 1, 35, 2, 0)
```
![c1aa2a6a37867447b75d0a9cf4f035b](https://user-images.githubusercontent.com/97984680/181415287-06268518-b3e0-43d2-91fb-555e9f0da168.png)

```python
#Export the model
import pickle
with open('MedicalExpenditure','wb') as f:
    pickle.dump(LR,f)
import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
```

# Flask Server

## Server.py
```python

# from flask import Flask, request, jsonify
# import util
# app = Flask(__name__)


# @app.route('/get_region')
# def get_region():
#     response = jsonify({
#         'region':util.get_region()
#     })
#     response.headers.add('Access-Control-Allow-Origin','*')
#     return response

# @app.route('/predict_medical_charge',methods=['POST'])
# def predict_medical_charge():
#     age = int(request.form['age'])
#     sex = int(request.form['sex'])
#     bmi = int(request.form['bmi'])
#     children = int(request.form['children'])
#     smoker = int(request.form['smoker'])
#     region = request.form['region']

#     response = jsonify({
#         'estimated_price':util.get_estimatied_price(region,age,sex,bmi,children,smoker)
#     })
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     return response
# if __name__ == "__main__":
#     print('Starting Python Flask Sever For Medical Cost Prediction')
#     util.load_save_artifacts()
#     app.run()
```

## util.py
```
# import json
# import pickle
# import numpy as np
# __regions = None
# __data_columns = None
# __model = None

# def get_estimatied_price(region,age, sex, bmi, children, smoker):
#     try:
#         reg_index = __data_columns.index(region.lower())
#     except:
#         reg_index=-1

#     a = np.zeros(len(__data_columns))
#     a[0] = age
#     a[1] = sex
#     a[2] = bmi
#     a[3] = children
#     a[4] = smoker
#     if reg_index >= 0:
#         a[reg_index] = 1
#     return round(__model.predict([a])[0],2)

# def get_region():
#     return __regions

# def load_save_artifacts():
#     print("Loading save artifacts... start")
#     global __data_columns
#     global __regions

#     with open("./artifacts/columns.json",'r') as f:
#         __data_columns = json.load(f)['data_columns']
#         __regions = __data_columns[5:]

#     global __model
#     with open("./artifacts/MC.pickle",'rb') as f:
#         __model = pickle.load(f)
#         print("loading saved artifacts...done")


# if __name__ == '__main__':
#     load_save_artifacts()
#     print(get_region())
#     print(get_estimatied_price('southwest',26,1,27,3,1))
#     print(get_estimatied_price('southwest', 26, 1, 27, 3, 0))
```
Use postman test the funtion of the Flask Server

![e13a200d3c679e0d484fd12eb15fada](https://user-images.githubusercontent.com/97984680/181415835-cae26a31-d147-4940-9019-24e833c6345b.png)

![03a261e7166ccd6a7c247f28e753a2a](https://user-images.githubusercontent.com/97984680/181415838-d07a9280-cd8d-4481-8fb0-4157d8fccc64.png)

Run successfully!!

# Client

## app.js

```javascrpit
function getChildrenValue() {
  var uiChildren = document.getElementsByName("uiChildren");
  for(var i in uiChildren) {
    if(uiChildren[i].checked) {
        return parseInt(i);
    }
  }
  return -1; // Invalid Value
}

function getSmokeValue() {
  var uiSmoke = document.getElementsByName("uiSmoke");
  for(var i in uiSmoke) {
    if(uiSmoke[i].checked) {
        return parseInt(i);
    }
  }
  return -1; // Invalid Value
}

function getGenderValue() {
  var uiGender = document.getElementsByName("uiGender");
  for(var i in uiGender) {
    if(uiGender[i].checked) {
        return parseInt(i);
    }
  }
  return -1; // Invalid Value
}

function onClickedEstimatePrice() {
  console.log("Estimate price button clicked");
 var age = document.getElementById("uiAge");
  var bmi = document.getElementById("uiBmi");
  var gender = getGenderValue();
  var smoke = getSmokeValue();
  var children = getChildrenValue();
  var region = document.getElementById("uiRegion");
  var estPrice = document.getElementById("uiEstimatedPrice");

  var url = "http://127.0.0.1:5000/predict_medical_charge";
  //var url = "/api/predict_medical_charge"

  $.post(url, {
      region: region.value,
      age: age.value,
      sex: gender,
      bmi: bmi.value,
      children: children,
      smoker: smoke,
      
  },function(data, status) {
      console.log(data.estimated_price);
      estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " USD</h2>";
      console.log(status);
  });
}


function onPageLoad() {
  console.log( "document loaded" );
  var url = "http://127.0.0.1:5000/get_region";
  //var url = "/api/get_region"
  $.get(url,function(data, status) {
      console.log("got response for get_location_names request");
      if(data) {
          var region = data.region;
          var uiRegion = document.getElementById("uiRegion");
          $('#uiRegion').empty();
          for(var i in region) {
              var opt = new Option(region[i]);
              $('#uiRegion').append(opt);
          }
      }
  });
}

window.onload = onPageLoad;
  
```

## app.css
```css
@import url(https://fonts.googleapis.com/css?family=Roboto:300);

.switch-field {
	display: flex;
	margin-bottom: 36px;
	overflow: hidden;
}

.switch-field input {
	position: absolute !important;
	clip: rect(0, 0, 0, 0);
	height: 1px;
	width: 1px;
	border: 0;
	overflow: hidden;
}

.switch-field label {
	background-color: #e4e4e4;
	color: rgba(0, 0, 0, 0.6);
	font-size: 14px;
	line-height: 1;
	text-align: center;
	padding: 8px 16px;
	margin-right: -1px;
	border: 1px solid rgba(0, 0, 0, 0.2);
	box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3), 0 1px rgba(255, 255, 255, 0.1);
	transition: all 0.1s ease-in-out;
}

.switch-field label:hover {
	cursor: pointer;
}

.switch-field input:checked + label {
	background-color: #a5dc86;
	box-shadow: none;
}

.switch-field label:first-of-type {
	border-radius: 4px 0 0 4px;
}

.switch-field label:last-of-type {
	border-radius: 0 4px 4px 0;
}

.form {
	max-width: 270px;
	font-family: "Lucida Grande", Tahoma, Verdana, sans-serif;
	font-weight: normal;
	line-height: 1.625;
	margin: 8px auto;
	padding-left: 16px;
	z-index: 2;
}

h2 {
	font-size: 18px;
	margin-bottom: 8px;
}
.age{
  font-family: "Roboto", sans-serif;
  outline: 0;
  background: #f2f2f2;
  width: 76%;
  border: 0;
  margin: 0 0 10px;
  padding: 10px;
  box-sizing: border-box;
  font-size: 15px;
  height: 35px;
  border-radius: 5px;
}

.bmi{
    font-family: "Roboto", sans-serif;
    outline: 0;
    background: #f2f2f2;
    width: 76%;
    border: 0;
    margin: 0 0 10px;
    padding: 10px;
    box-sizing: border-box;
    font-size: 15px;
    height: 35px;
    border-radius: 5px;
  }

.Region{
  font-family: "Roboto", sans-serif;
  outline: 0;
  background: #f2f2f2;
  width: 76%;
  border: 0;
  margin: 0 0 10px;
  padding: 10px;
  box-sizing: border-box;
  font-size: 15px;
  height: 40px;
  border-radius: 5px;
}

.submit{
  background: #a5dc86;
  width: 76%;
  border: 0;
  margin: 25px 0 10px;
  box-sizing: border-box;
  font-size: 15px;
	height: 35px;
	text-align: center;
	border-radius: 5px;
}

.result{
		background: #c1cac5;
		width: 76%;
		border: 0;
		margin: 25px 0 10px;
		box-sizing: border-box;
		font-size: 15px;
		height: 35px;
		text-align: center;
}

.img {
  background: url('https://th.bing.com/th/id/OIP.rsmcC4SIEIzVLyvsPInKawHaE7?pid=ImgDet&rs=1');
	background-repeat: no-repeat;
  background-size: auto;
  background-size:100% 100%;
  -webkit-filter: blur(5px);
  -moz-filter: blur(5px);
  -o-filter: blur(5px);
  -ms-filter: blur(5px);
  filter: blur(15px);
  position: fixed;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  z-index: -1;
}

body, html {
  height: 100%;
}
```

## app.HTML
```HTML
<!DOCTYPE html>
<html>
<head>
    <title>Medical Cost Prediction</title>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <script src="app.js"></script>
	<link rel="stylesheet" href="app.css">
</head>
<body>
<div class="img"></div>
<form class="form">
	<h2>Age</h2>
    <input class="age"  type="text" id="uiAge" class="floatLabel" name="Age" value="21">

	<h2>Bmi</h2>
    <input class="bmi"  type="text" id="uiBmi" class="floatLabel" name="Bmi" value="25">

	<h2>Gender</h2>
	<div class="switch-field">
		<input type="radio" id="radio-Gender-1" name="uiGender" value="0" checked/>
		<label for="radio-Gender-1">Female</label>
		<input type="radio" id="radio-Gender-2" name="uiGender" value="1"/>
		<label for="radio-Gender-2">Male</label>
	</div>
	
	</form>

<form class="form">
	<h2>Do You Smoke?</h2>
	<div class="switch-field">
		<input type="radio" id="radio-Smoke-1" name="uiSmoke" value="0" checked/>
		<label for="radio-Smoke-1">No</label>
		<input type="radio" id="radio-Smoke-2" name="uiSmoke" value="1"/>
		<label for="radio-Smoke-2">Yes</label>
	</div>

	<h2>How Many Children Do You have?</h2>
	<div class="switch-field">
		<input type="radio" id="radio-Children-1" name="uiChildren" value="0"/>
		<label for="radio-Children-1">0</label>
		<input type="radio" id="radio-Children-2" name="uiChildren" value="1" checked/>
		<label for="radio-Children-2">1</label>
		<input type="radio" id="radio-Children-3" name="uiChildren" value="2"/>
		<label for="radio-Children-3">2</label>
		<input type="radio" id="radio-Children-4" name="uiChildren" value="3"/>
		<label for="radio-Children-4">3</label>
		<input type="radio" id="radio-Children-5" name="uiChildren" value="4"/>
		<label for="radio-Children-5">4</label>
	</div>

		<h2>Region</h2>
	<div>
  <select class="Region" name="" id="uiRegion">
    <option value="" disabled="disabled" selected="selected">Choose a region</option>
  </select>
</div>
	<button class="submit" onclick="onClickedEstimatePrice()" type="button">Estimate Cost</button>
	<div id="uiEstimatedPrice" class="result">	<h2></h2> </div>
</body>
</html>
```
# Test Run
![60f814c21041fa8fee8e1b4135c664c](https://user-images.githubusercontent.com/97984680/181418302-9775f3da-d756-41d8-a5dc-b5b4e43bf2a3.png)

![aa80d5b92228f5d4a7c3d84f31fa5d6](https://user-images.githubusercontent.com/97984680/181418318-6cb479d5-26de-4272-aaca-33c4a3dc695c.png)

![96ea04e3dae62ca51bba501bd44d555](https://user-images.githubusercontent.com/97984680/181418325-590b9c5f-568a-4816-ae1e-1d239ce36f9d.png)



