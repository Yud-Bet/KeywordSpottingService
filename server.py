import random
import os
from flask import Flask, request
from flask.json import jsonify
from keyword_spotting_service import Keyword_Spotting_Service

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    kss = Keyword_Spotting_Service()
    predicted_keyword =  kss.predict(file_name)

    os.remove(file_name)

    data = {'keyword': predicted_keyword}
    return jsonify(data)

@app.route('/', methods=['GET'])
def welcome():
    return 'This is keyword spotting service'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
