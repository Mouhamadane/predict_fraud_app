from flask import Flask, request, jsonify
from model.ml_param import predict
import pickle

app = Flask('fraud_prediction')


@app.route('/', methods=['GET'])
def home():
    return "My app is working..."


@app.route('/predict', methods=['POST'])
def prediction():
    data = request.get_json()
    with open('model/classifier.pkl', 'rb') as file:
        classifier = pickle.load(file)
        file.close()
    prediction = predict(data, classifier)
    response = {'fraud': int(prediction)}
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
