from flask import Flask
from flask import jsonify,request
import pickle

app = Flask(__name__)

with open('dv.bin', 'rb') as dict_in:
    dict_vectorizer = pickle.load(dict_in)
    # print("dict file opened")
    with open('model1.bin', 'rb') as model_in:  
        model = pickle.load(model_in)

def predict_value(client,dv,model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]
    
@app.route("/predict",methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_value(customer,dict_vectorizer,model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction), ## we need to conver numpy data into python data in flask framework
        'churn': bool(churn),  ## same as the line above, converting the data using bool method
    }

    return jsonify(result)  ## send back the data in json format to the user

    

if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
