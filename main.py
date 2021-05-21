from flask import Flask, request, jsonify
from model.ml_param import predict
from flask_restplus import Api, Resource, fields

import pickle

app = Flask('fraud_prediction')
api = Api(app=app, version='1.0', title='Fraud Detection Api', description='', validate=True)
pred = api.namespace('Prediction', description="Make a prediction of a transaction")
data_pred = api.model('Predict Informations', {
    'step': fields.Float(required=True),
    'category': fields.Float(required=True),
    'customer': fields.String(required=True),
    'gender': fields.Float(required=True),
    'age': fields.Float(required=True),
    'amount': fields.Float(required=True),
    'merchant': fields.String(required=True)
})


def getdata():
    data = request.get_json()
    step = data.get('step')
    customer = data.get('customer')
    gender = data.get('gender')
    cat_age = data.get('age')
    category = data.get('category')
    amount = data.get('amount')
    merchant = data.get('merchant')

    df = {
        'step': step,
        'customer': customer,
        'gender': gender,
        'age': cat_age,
        'category': category,
        'amount': amount,
        'merchant': merchant
    }
    return df


@pred.route('/pred')
class GetHome(Resource):
    @pred.response(200, 'Home : Success')
    def get(self):
        """
        Home Page
        """

    @api.expect(data_pred)
    @pred.response(200, 'Prediction : Success')
    @pred.response(400, 'Error : Validation error')
    def post(self):
        """
        Make a prediction
        """
        data = request.get_json()
        with open('model/classifier.pkl', 'rb') as file:
            classifier = pickle.load(file)
            file.close()
        prediction = predict(data, classifier)
        response = {'fraud': int(prediction)}
        return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
