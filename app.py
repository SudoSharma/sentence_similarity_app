"""Creates a flask application to check the similarity of two user queries."""

from evaluate import load_model, evaluate
import dill as pickle
from flask import Flask
from flask_restful import reqparse, Api, Resource
import torch
from model.utils import AppData
import numpy as np
import boto3
import os

app = Flask(__name__)
api = Api(app)

# Store s3 data. Comment out this section until 'model_args' if not on heroku
S3_BUCKET = os.environ.get('S3_BUCKET')
files = [
    'data/args.pkl', 'data/TEXT.pkl',
    'saved_models/bimpm_quora_17_04_11.pt'
]
s3 = boto3.resource('s3')
for s3_file in files:
    s3.Object(S3_BUCKET, s3_file).download_file(s3_file)

model_args = pickle.load(open('./data/args.pkl', 'rb'))
model_args.device = torch.device('cuda:0' if torch.cuda.
                                 is_available() else 'cpu')

# Load arguments, q1 and q2 for sentence 1 and 2
parser = reqparse.RequestParser()
parser.add_argument('q1', required=True, help="Sentence 1 cannot be blank!")
parser.add_argument('q2', required=True, help="Sentence 2 cannot be blank!")


class PredictSentenceSimilarity(Resource):
    """An endpoint for the RESTful API, inherits form the Resource class, and
    loads model, evaluates queries, and returns a probability distribution over
    the output classes.

    """

    def get(self):
        """Specify a GET HTTP method for model predictions.

        Returns
        -------
        json
            A json file containing the queries, predictions, and probabilities.

        """
        # Parse args and prep app_data
        app_args = parser.parse_args()
        q1, q2 = app_args['q1'], app_args['q2']
        app_data = [q1, q2]

        model_data = AppData(model_args, app_data)
        model = load_model(model_args, model_data)  # Load model params

        # Calculate probability distribution of classes
        preds = evaluate(model, model_args, model_data).numpy()

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        sm_preds = softmax(preds).tolist()
        prediction = 'similar' if max(
            sm_preds) == sm_preds[1] else 'not similar'
        preds = preds.tolist()

        return {
            'query_one': q1,
            'query_two': q2,
            'neg_layer_output': round(preds[0], 4),
            'pos_layer_output': round(preds[1], 4),
            'neg_probability': round(sm_preds[0], 4),
            'pos_probability': round(sm_preds[1], 4),
            'prediction': prediction
        }


api.add_resource(PredictSentenceSimilarity, '/')  # Add endpoint

if __name__ == '__main__':
    app.run(debug=True)  # Only executed when script is run directly
