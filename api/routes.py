from flask import request, jsonify
from api import api
from api.validator import validate_predict_request
from src.predict import predict_depression

@api.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Validate incoming input
        valid, result = validate_predict_request(data)
        if not valid:
            return jsonify(
                { 
                    "data":"",
                    "message":result, 
                    "error":True
                }
            ), 400

        text = result
        # Run prediction
        prediction = predict_depression(text)
        print(prediction)

        return jsonify(  
            {
                "message":"Prediction successful",
                "data":prediction,
                "error":False
            }
        ), 200

    except Exception as e:
        return jsonify(
            { 
                "message": "Internal server error",
                "data": str(e),
                "error": True
            }
        ), 500
