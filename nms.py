import cv2
import numpy as np
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

classNames= []
classFile ='coco.names'
with open(classFile,'rt') as f:
   classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)&amp;#91;0])
    confs = list(map(float,confs))
    #print(type(confs&amp;#91;0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    #print(indices)

    for i in indices:
        i = i&amp;#91;0]
        box = bbox&amp;#91;i]
        x,y,w,h = box&amp;#91;0],box&amp;#91;1],box&amp;#91;2],box&amp;#91;3]
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img,classNames&amp;#91;classIds&amp;#91;i]&amp;#91;0]-1].upper(),(box&amp;#91;0]+10,box&amp;#91;1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("OUTPUT",img)
    cv2.waitKey(1)