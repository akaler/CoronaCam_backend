# flask and other web required frameworks
from flask import Flask, jsonify, request, Response, make_response
from flask_cors import CORS
import json
import numpy as np
#import argparse

# from app import violations
import server_social_distance_detector 
import server_detect_mask 
app = Flask(__name__)
CORS(app)

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
        "current_violations_sd": server_social_distance_detector.current_violations_sd,
        "average_violations_sd": server_social_distance_detector.average_violations_sd,
        "current_violations_masks": server_detect_mask.current_violations_masks,
        "average_violations_masks": server_detect_mask.average_violations_masks
    }
    sentenceJSON = json.dumps(sentenceJSON, default=serialize_sets)
    jsonData.append(sentenceJSON)
    return jsonify({
        "data": jsonData,
    })

@app.route('/video_social_distance')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(server_social_distance_detector.gen_social_distancing(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_mask')
def video_mask():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(server_detect_mask.gen_mask(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
