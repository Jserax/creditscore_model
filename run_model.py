import joblib
import pandas as pd
import json
from flask import Flask, request


model = joblib.load('models/model.joblib')
pred_dict = {0: 'Poor', 1: 'Standard', 2: 'Good'}

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def index():
    dict_request = json.loads(request.get_json())
    df = pd.DataFrame(dict_request, index=[0])
    prediction = model.predict(df).item()
    return pred_dict[prediction]

if __name__ == '__main__':
    app.run(host="0.0.0.0:5000", debug=False)