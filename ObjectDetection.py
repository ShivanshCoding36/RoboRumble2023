import cv2
import matplotlib.pyplot as mat
from time import sleep

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model, config_file)
Bio=["person","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe""banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake"]
ClassLabels = []
fileName = "labels.txt"
with open(fileName, "rt") as fo:
    ClassLabels = fo.read().rstrip("\n").split("\n")

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
cap.set(3,720)#Width
cap.set(4,1080)#Height
cap.set(10,100)#Brightness

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open the video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    success, img = cap.read()
    ClassIndex, Confidence, bbox = model.detect(img, confThreshold=0.5)
    print(ClassIndex)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
            if ClassInd - 1 < len(ClassLabels):
                label = ClassLabels[ClassInd - 1]
            else:
                label = 'Unknown'
            if label in Bio:
                type="BioDegradable"
            else:
                type="Non-BioDegradable"
            cv2.rectangle(img, boxes, (255, 0, 0), 2)
            cv2.putText(
                img,
                type,
                (boxes[0] + 10, boxes[1] + 40),
                font,
                fontScale=font_scale,
                color=(0, 255, 0),
                thickness=3,
            )
        cv2.imshow("Video", img)
            


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
