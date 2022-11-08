import cv2
import jsonpickle
import numpy as np
from flask import Flask, request, Response

from code_from_IA.predict_recomendation_IA import Pipeline_IA

app = Flask(__name__)

@app.route('/detect_objets', methods=["POST"])
def detect_objets():
    #jsonEnviado = request.get_json()
    pass

@app.route('/predict_recomendation', methods=["POST"])
def predict_recomendation():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    response_pickled = jsonpickle.encode(img)
    return Response(response=response_pickled, status=200, mimetype="application/json")
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')