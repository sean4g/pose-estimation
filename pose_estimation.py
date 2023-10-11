from ultralytics import YOLO
from pupil_apriltags import Detector
from matplotlib import animation as animation, pyplot as plt
from functools import partial
import cv2
import numpy as np

BASIS = [(718.75, 399.34), (707.88, 445.79)]

'''
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16
'''

def get_keypoints(results):
    li = []
    if len(results[0].boxes.conf) > 1:
        li.append(results[0].keypoints.xy[1][9])
        li.append(results[0].keypoints.xy[1][10])
    else:
        li.append(results[0].keypoints.xy[0][9])
        li.append(results[0].keypoints.xy[0][10])
    return li

# initialize apriltag detector
detector = Detector(families="tag16h5", debug=1)
image = cv2.imread("test_image_multiple_02.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

model = YOLO("YOLOv8x-pose-p6.pt")
source = "pose_video.mp4"
# results = model.predict(source, save=True, conf=0.65, boxes=False)
# results = model.predict("test.jpg", save=True, imgsz=320)
# print(results[0].keypoints)

# initialize feed
cap = cv2.VideoCapture(source)

#initialize graph
plt.ion()
fig, ax = plt.subplots(figsize =(16, 9))
tags = np.array(['left', 'right', 'distance'])
y = np.array([750, 750, 1])
graph = ax.bar(tags, y)
plt.show(block=False)



while cap.isOpened():
    success, frame = cap.read()
    if success:
        # get results
        results = model.predict(frame, save=True, imgsz=320, conf=0.65, boxes=False)

        # convert to gray for apriltags
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apriltag_det = detector.detect(gray)
        # print(apriltag_det)

        # get left/right wrist points
        annotated_frame = results[0].plot()

        #if person found
        if results[0].keypoints.has_visible:
            left, right = get_keypoints(results)
            show = [left[0].item(), right[0].item(), BASIS[0][0] - max(left[0].item(), right[0].item())]
        else:
            show = [0, 0, BASIS[0][0]]

        cv2.imshow("Test", annotated_frame)

        # Show Plot
        for rect, h in zip(graph,show):
            rect.set_height(h)
        fig.canvas.draw()
        fig.canvas.flush_events()        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


# cap.release()
# cv2.destroyAllWindows()
