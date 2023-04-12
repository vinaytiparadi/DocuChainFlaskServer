import pickle
import numpy as np
from statistics import mode
from flask import Flask, jsonify

app = Flask(__name__)

# load the trained model
with open('final_rf_model.pkl', 'rb') as f:
    final_rf_model = pickle.load(f)

with open('final_nb_model.pkl', 'rb') as f:
    final_nb_model = pickle.load(f)

with open('final_svm_model.pkl', 'rb') as f:
    final_svm_model = pickle.load(f)

# load the symptom index dictionary and predictions classes
with open('data_dict.pkl', 'rb') as f:
    data_dict = pickle.load(f)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/predict/<string:s>')
def predict(s):
    input_data = []
    for char in s:
        input_data.append(int(char))
    
    # reshape the input data and convert it into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }

    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)