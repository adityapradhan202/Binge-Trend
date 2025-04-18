from flask import Flask, request, jsonify
from flask_cors import CORS 
# pip install flask-cors

from fusionator_v0 import fusionator_v0

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    synopsis = data.get('synopsis', '')

    # Using the fusionator_v1 we made in ml_backend directory
    result = fusionator_v0(text=synopsis)
    return jsonify(result) # Returns prediction probability of all the genres here (in json)

if __name__ == "__main__":
    app.run(debug=True)
