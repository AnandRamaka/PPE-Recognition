import logging, thread, threading
import time, math, argparse, imutils, shutil
import pickle, os, cv2, io
import copy, boto3, json
import pafy, string, random, glob
from threading import Lock, Thread
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import keras
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from botocore.exceptions import ClientError

def iob( r1, r2):
    x_left = max(r1[0], r2[0])
    y_top = max(r1[1], r2[1])
    x_right = min(r1[2], r2[2])
    y_bottom = min(r1[3], r2[3])
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
    bb2_area = (r2[2] - r2[0]) * (r2[3] - r2[1])
    num = (intersection_area / bb1_area, intersection_area / bb2_area)
    return (intersection_area / bb1_area, intersection_area / bb2_area)

def is_inside(r1, r2):
    if r1[0] > r2[0] and r1[1] > r2[1] and r1[2] < r2[2] and r1[3] < r2[3]:
        return True
    else:
        return False

def iou( r1, r2 ):
	x_left = max(r1[0], r2[0])
	y_top = max(r1[1], r2[1])
	x_right = min(r1[2], r2[2])
	y_bottom = min(r1[3], r2[3])
	if x_right < x_left or y_bottom < y_top:
		return 0.0
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	bb1_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
	bb2_area = (r2[2] - r2[0]) * (r2[3] - r2[1])
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	return iou

def iou( r1, r2 ):
	x_left = max(r1[0], r2[0])
	y_top = max(r1[1], r2[1])
	x_right = min(r1[2], r2[2])
	y_bottom = min(r1[3], r2[3])
	if x_right < x_left or y_bottom < y_top:
		return 0.0
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	bb1_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
	bb2_area = (r2[2] - r2[0]) * (r2[3] - r2[1])
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	return iou

def detect_PPE_helper(photo):
    client=boto3.client('rekognition')
    if cv2.imread("./frame.jpg") is None:
        return
    with open("./frame.jpg", "rb") as image:
        try:    
            response = client.detect_protective_equipment(Image={'Bytes': image.read()})
        except botocore.exceptions.ClientError:
            return

    mask_check = []
    if not response['Persons']:
        return ([], False)
    for person in response['Persons']:
        bbox = person["BoundingBox"]
        body_parts = person['BodyParts']
        for body_part in body_parts:
            if body_part['Name'] != 'FACE':
                continue
            ppe_items = body_part['EquipmentDetections']
            if len(ppe_items) == 0:
                #print ('No PPE detected on ' + body_part['Name'])
                mask_check.append( (bbox, False) )
            else:
                for ppe_item in ppe_items:
                    if ppe_item['Type'] == 'FACE_COVER':
                        mask_check.append( (ppe_item['BoundingBox'], True) )
                    #print('\t\t' + ppe_item['Type'] + '\n\t\t\tConfidence: ' + str(ppe_item['Confidence'])) 
                    # print('\t\tCovers body part: ' + str(ppe_item['CoversBodyPart']['Value']) + '\n\t\t\tConfidence: ' + str(ppe_item['CoversBodyPart']['Confidence']))
    #print("Checking for masks...")
    print(mask_check)
    return mask_check

def detect_PPE_helper2(photo):
    client=boto3.client('rekognition')
    if cv2.imread("./frame.jpg") is None:
        return
    with open("./frame.jpg", "rb") as image:
        try:    
            response = client.detect_protective_equipment(Image={'Bytes': image.read()})
        except ClientError as e:
            print("AWS Client Error")
            return

    mask_check = []
    if not response['Persons']:
        return ([], False)
    for person in response['Persons']:
        temp = []
        bbox = person["BoundingBox"]
        body_parts = person['BodyParts']
        totalPPE = 0
        found_face_cover = False
        for body_part in body_parts:
            ppe_items = body_part['EquipmentDetections']
            for ppe_item in ppe_items:
                temp.append(ppe_item['Type'])
                totalPPE += 1
                if ppe_item['Type'] == 'FACE_COVER':
                    found_face_cover = True
                    bbox = ppe_item['BoundingBox']
        
        if totalPPE == 0:    
            mask_check.append( (bbox, False) )
        
        if found_face_cover:
            mask_check.append( [bbox, True] + temp )

    #print("Checking for masks...")
    #print(mask_check)
    return mask_check

def detect_PPE(lock):
    global detect_now, frame
    while True:
        if not detect_now or frame is None: continue
        mask_check = detect_PPE_helper2('./frame.jpg')
        #print(mask_check)
        pickle.dump(mask_check, open("./mask_check.pickle", "wb"))

#detect_PPE_helper('./NoPPE1.jpg')

print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model",
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

detect_now = False
lock = Lock()
t = Thread(target=detect_PPE, daemon=True, args=(lock,))
t.start()

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
stream = io.BytesIO()
frame_people = {}
new_people = {}
mask_check = []
color_dict = { "Red" : (0, 0, 255), "Orange" : (0,215,255), "Green" : (0, 255, 0) }
pickle.dump(mask_check, open("mask_check.pickle", "wb"))

while True:
    detect_now = False
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    cv2.imwrite('./frame.jpg', frame)
    detect_now = True
    imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()
    new_people = {}
    mask_check = pickle.load(open(r"mask_check.pickle", "rb"))
    #print(mask_check)
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > .5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            ppe_list = []
            found = False
            for key, val in frame_people.items():
                if iou( key, (startX, startY, endX, endY) ) > .5:
                    found = True
                    new_people.update( {(startX, startY, endX, endY) : val} )
            if not found:
                new_people.update( {(startX, startY, endX, endY) : "Orange"} )

            print(mask_check)
            if not mask_check: continue
            for m in mask_check:
                if not m: continue
                tempX = int(m[0]["Left"] * w)
                tempY = int(m[0]["Top"] * h )
                coords = ( tempX, tempY, int(tempX + m[0]["Width"] * w), int(tempY + m[0]["Height"] * h) )
                #coords2 = (startX, startY, endX, endY)
                if 'HAND_COVER' in m:
                    print("Hand cover found!!!!")
                    #print(iob_num)
                iob_num = iob(coords, (startX, startY, endX, endY))
                #print(iob_num)

                if not iob_num: continue

                if iob_num[0] < .5 and iob_num[1] < .5:
                    continue
                if m[1]:
                    ppe_list = m[2:]
                    ppe_str = ", ".join(ppe_list)
                    #print(ppe_list, ppe_str)
                    new_people[(startX, startY, endX, endY)] = ppe_str
                else:
                    new_people[(startX, startY, endX, endY)] = "Red"

    #print(new_people, mask_check)
    for key, val in new_people.items():
        color = val if val == "Red" or val == "Orange" else "Green"
        cv2.rectangle(frame, key[0:2], key[2:4], color_dict[color], 2)
        if color == "Green":
            y = key[1] - 10 if key[1] - 10 > 10 else key[1] + 10
            cv2.putText(frame, val, (key[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_dict[color], 2)

    frame_people = copy.deepcopy(new_people)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
	