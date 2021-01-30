# flask and other web required frameworks
from flask import Flask, jsonify, request, Response, make_response
from flask_cors import CORS
import json
import numpy as np
#import argparse

# from app import violations
from server_social_distance_detector import gen_social_distancing 
from server_detect_mask import gen_mask
app = Flask(__name__)
CORS(app)

violations = 1
# # #
# # # # ROUTES        
# # # # # # # # #
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/getViolations", methods=['GET'])
def getViolations():
    # filee = request.files['file'] # get file from request body
    jsonData = []
    sentenceJSON = {
        "violations": violations,
    }
    sentenceJSON = json.dumps(sentenceJSON, default=serialize_sets)
    jsonData.append(sentenceJSON)
    return jsonify({
        "data": jsonData,
    })

@app.route('/video_social_distance')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_social_distancing(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_mask')
def video_mask():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_mask(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
