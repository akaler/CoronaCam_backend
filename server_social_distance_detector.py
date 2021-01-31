import json
from random import randint
from pyimagesearch import social_distancing_config as config
import video_input_config as video_config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os


#global current_violations_sd  
#global average_violations_sd 


def gen_social_distancing():
    global current_violations_sd   
    global average_violations_sd  
    global violations_in_past 

    current_violations_sd   =0
    average_violations_sd   =0
    violations_in_past = []
    #for i in violations_in_past:
    #    i = 0
    args = {"input": video_config.SOCIAL_DISTANCE_INPUT, "output": video_config.SOCIAL_DISTANCE_OUTPUT, "display": 1}
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
    vs = cv2.VideoCapture(args["input"])
    # vs.set(cv2.CV_CAP_PROP_FPS, 15)
    vs.set(cv2.FONT_HERSHEY_SIMPLEX, 10)
    writer = None
    # loop over the frames from the video stream
    counter = 0
    counter20 = 0
    total_violations = 0 
    while True:
        # read the next frame from the file
        counter = counter + 1
        if(counter > 10000):
            counter = 1
            counter20 = 0
            
        (grabbed, frame) = vs.read()

        if(counter % 20 != 0):
            continue
        counter20 += 1
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
        current_violations_sd = len(violate)
        total_violations += current_violations_sd
        average_violations_sd = (total_violations)/float(counter20)
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