from ultralytics import YOLO
from pupil_apriltags import Detector
from matplotlib import animation as animation, pyplot as plt
from functools import partial
import cv2
import numpy as np

# to edit excel
import pandas as pd

# BASIS = [(718.75, 399.34), (707.88, 445.79)]

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

df = pd.DataFrame({'Frame': [], 'Nose': [], 'Left Eye': [], 'Right Eye': [], 'Left Ear': [], 'Right Ear': [],
                   'Left Shoulder': [], 'Right Shoulder': [], 'Left Elbow': [], 'Right Elbow': [],
                   'Left Wrist': [], 'Right Wrist': [], 'Left Hip': [], 'Right Hip': [], 
                   'Left Knee': [], 'Right Knee': [], 'Left Ankle': [], 'Right Ankle': []})

def get_keypoints(results):
    if len(results[0].boxes.conf) > 1:
        return results[0].keypoints.xy[1].tolist()

    return results[0].keypoints.xy[0].tolist()

# initialize apriltag detector
detector = Detector(families="tag16h5", debug=1)

model = YOLO("YOLOv8n-pose.pt")
source = "20230529114514VCAP1_crop_trim.mp4"
# results = model.predict(source, save=True, imgsz=320, conf=0.5, boxes=True)
cap = cv2.VideoCapture(source)

'''#initialize graph
plt.ion()
fig, ax = plt.subplots(figsize =(16, 9))
tags = np.array(['left', 'right', 'distance'])
y = np.array([750, 750, 1])
graph = ax.bar(tags, y)
plt.show(block=False)'''

# data editor
frame_num = 0

keypoints=None

while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame_num += 1
        # get results
        results = model.predict(frame, save=False, imgsz=320, conf=0.5, boxes=False)

        # convert to gray for apriltags
        '''# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apriltag_det = detector.detect(gray)
        # print(apriltag_det)'''

        # get left/right wrist points
        annotated_frame = results[0].plot()

        #if person found
        new_row = [frame_num] + ['N/A'] * 17
        
        if results[0].keypoints.has_visible:
            keypoints = get_keypoints(results)
            new_row = [frame_num] + keypoints
            # show = [left[0].item(), right[0].item(), BASIS[0][0] - max(left[0].item(), right[0].item())]
            

        # cv2.imshow("Test", annotated_frame)

        '''# Show Plot
        # for rect, h in zip(graph,show):
        #     rect.set_height(h)
        # fig.canvas.draw()
        # fig.canvas.flush_events()'''

        df.loc[len(df.index)] = new_row    

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    
if keypoints:
    BASIS = keypoints[10]

distances = []
frames = []

for row in df.itertuples(index=False):
    if row[11] != 'N/A':
        distance = BASIS[0] - row[11][0]
        frames.append(row[0])
        distances.append(distance if distance >= 0 else 0)



plt.plot(frames, distances)
plt.show()


df.to_csv(f'out_{source[:-4]}.csv', index=False)

# cap.release()
# cv2.destroyAllWindows()
