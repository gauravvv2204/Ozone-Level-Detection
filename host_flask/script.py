import numpy as np
import pickle
import pandas as pd 
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect, url_for
from forms import OzoneForm
app= Flask(__name__)

app.config['SECRET_KEY'] = '69420'

filename = r'D:\College HW\ML\Project\Ozone-Level-Detection\Models'
model = r'{}\RandomForest.sav'.format(filename)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 72)
    loaded_model = pickle.load(open(model, "rb"))
    # if(model.find("Logistic")):
    #     pca_reload = pickle.load(open(r"{}\LogisticPCA.pkl".format(filename),'rb'))
    #     to_predict = pca_reload.transform(to_predict)
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route('/', methods= ['GET','POST'])
def home():
    form = OzoneForm()
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        new_form =[]
        for l in to_predict_list:
            print(l)
        for i in range(1,73):
            new_form.append(float(to_predict_list[i]))
        result = ValuePredictor(new_form)      
        if (result > 0.5):
            prediction ='Its an Ozone Day'
        else:
            prediction ='Its not an Ozone Day'           
        return render_template("result.html", prediction = prediction)
    return render_template('home.html', form = form)

if __name__ == '__main__':
    app.run(debug=True)