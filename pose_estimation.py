from ultralytics import YOLO
from pupil_apriltags import Detector
from matplotlib import animation as animation, pyplot as plt
from functools import partial
import cv2
import numpy as np

# to edit excel
import pandas as pd

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
        li.append(results[0].keypoints.xy[1][9].tolist())
        li.append(results[0].keypoints.xy[1][10].tolist())
        li.append(results[0].keypoints.xy[1][2].tolist())
        li.append(results[0].keypoints.xy[1][4].tolist())
        li.append(results[0].keypoints.xy[1][6].tolist())
    else:
        li.append(results[0].keypoints.xy[0][9].tolist())
        li.append(results[0].keypoints.xy[0][10].tolist())
        li.append(results[0].keypoints.xy[0][2].tolist())
        li.append(results[0].keypoints.xy[0][4].tolist())
        li.append(results[0].keypoints.xy[0][6].tolist())
    return li

# initialize apriltag detector
detector = Detector(families="tag16h5", debug=1)

model = YOLO("YOLOv8n-pose.pt")
source = "pose_video.mp4"
# results = model.predict(source, save=False, imgsz=320, conf=0.65, boxes=False)
cap = cv2.VideoCapture(source)

#initialize graph
# plt.ion()
# fig, ax = plt.subplots(figsize =(16, 9))
# tags = np.array(['left', 'right', 'distance'])
# y = np.array([750, 750, 1])
# graph = ax.bar(tags, y)
# plt.show(block=False)

# data editor
df = pd.DataFrame({'Frame': [], 'Left Wrist': [], 'Right Wrist': [], 'Right Eye': [], 'Right Ear': [],  'Right Shoulder': []})
frame_num = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame_num += 1
        # get results
        results = model.predict(frame, save=False, imgsz=320, conf=0.65, boxes=False)

        # convert to gray for apriltags
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apriltag_det = detector.detect(gray)
        # print(apriltag_det)

        # get left/right wrist points
        annotated_frame = results[0].plot()

        #if person found
        new_row = [frame_num, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
        
        if results[0].keypoints.has_visible:
            keypoints = get_keypoints(results)
            new_row = [frame_num] + keypoints
            # show = [left[0].item(), right[0].item(), BASIS[0][0] - max(left[0].item(), right[0].item())]
            

        # cv2.imshow("Test", annotated_frame)

        # Show Plot
        # for rect, h in zip(graph,show):
        #     rect.set_height(h)
        # fig.canvas.draw()
        # fig.canvas.flush_events()

        df.loc[len(df.index)] = new_row    

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

print(df.size)
df.to_csv(f'out_{source[:-4]}.csv', index=False)

# cap.release()
# cv2.destroyAllWindows()
