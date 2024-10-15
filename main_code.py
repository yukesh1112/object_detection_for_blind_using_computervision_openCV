import cv2
import pyttsx3

# Threshold to detect object
thres = 0.45  

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set video frame width and height
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class names from the coco.names file
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize text-to-speech engine
t_s = pyttsx3.init()
previous_classId = None  # To track the previously detected object

while True:
    # Read the frame from the camera
    success, img = cap.read()
    if not success:
        print("Error: Failed to read frame.")
        break

    # Detect objects in the image
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Speak the object name if it has changed
            if previous_classId != classId:
                obs = str(classNames[classId - 1])
                t_s.say(obs)
                t_s.runAndWait()
                previous_classId = classId
    else:
        previous_classId = None  # Reset if no objects detected

    # Show the output image
    cv2.imshow("Output", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
