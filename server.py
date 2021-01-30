# flask and other web required frameworks
from flask import Flask, jsonify, request, Response, make_response
from flask_cors import CORS
import json
from random import randint
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
#face-mask imports

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import time

# from app import violations

app = Flask(__name__)
CORS(app)

violations = 1
# # #
# # # # ROUTES        
# # # # # # # # #
def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

def gen_social_distancing():
    global violations
    args = {"input": "store.mp4", "output": "store_out.avi", "display": 1}
    #args["input"] = "store.mp4"
    #args["output"] = "store_out.avi"
    #args["display"] = 1
    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    #vs = cv2.VideoCapture(args["input"] if args["input"] else 0) # sami
    vs = cv2.VideoCapture("store.mp4")
    # vs.set(cv2.CV_CAP_PROP_FPS, 15)
    vs.set(cv2.FONT_HERSHEY_SIMPLEX, 10)
    writer = None
    # loop over the frames from the video stream
    counter = 1
    while True:
        # read the next frame from the file
        counter = counter + 1
        if(counter > 10000):
            counter = 1
            
        (grabbed, frame) = vs.read()

        if(counter % 20 != 0):
            continue
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=450)
	    # frame = np.dstack([frame, frame, frame])
        results = detect_people(frame, net, ln,
            personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        violate.add(i)
                        violate.add(j)
                        #detected = True



        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        # draw the total number of social distancing violations on the
        # output frame
        violations = violations + len(violate)
        text = "Social Distancing Violations: {}".format(len(violate))
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
        #yield(frame.tobytes())
        #frame = frame.tobytes()

        #yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

        # check to see if the output frame should be displayed to our
        # screen
        
        if args["display"] > 10:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def detect_and_predict_mask(frame, faceNet, maskNet):
    args = {"face": "face_detector", "model": "mask_detector.model", "confidence": 0.5}
	# grab the dimensions of the frame and then construct a blob
	# from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

# only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations 
    return (locs, preds)


def gen_mask():
    # construct the argument parser and parse the arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
        default="face_detector",
        help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    """
    args = {"face": "face_detector", "model": "mask_detector.model", "confidence": 0.5}
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    vs = cv2.VideoCapture("entry_cam.mp4")
    #time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        (grabbed, frame) = vs.read()
        #frame = imutils.resize(frame, width=400)
        if not grabbed:
            break
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    #print(frame)
    # show the output frame
    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    #if key == ord("q"):
    #	break

# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()

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

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_social_distancing(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_mask')
def video_mask():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_mask(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
