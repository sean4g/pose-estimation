import cv2
import time

# input video list 
video_files = ['20230814170153VCAP1.mp4', '20230601135829VCAP1.mp4', '20230529120140VCAP1.mp4']

# create 'Cropped' folder if there's none
if not cv2.utils.fs.exists('Cropped'):
    cv2.utils.fs.createDirectory('Cropped')

for video in video_files:
    start = time.time()

    # open video
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print(f"Error: Couldn't open the video {video}.")
        continue

    # get width, height, fps
    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # cropping values (lower left)
    x = 0
    y = h_frame // 2
    w = w_frame // 2
    h = h_frame // 2

    # output file name and path
    output_file = 'Cropped/' + video.split('/')[-1].replace('.mp4', '_cropped.mp4')
    
    # output video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # crop
        crop_frame = frame[y:y+h, x:x+w]
        out.write(crop_frame)

    cap.release()
    out.release()
    print(f'Processed {video} in {time.time()-start} seconds')
