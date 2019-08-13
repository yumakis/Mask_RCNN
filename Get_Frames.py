import numpy as np
import cv2
import sys
import os

"""Get the name of the video to extract"""

default = "/media/terence/DATA/Docs_for_test/VID_20170924_105126.mp4"

if len(sys.argv) < 2:
    video_name = default
else:
    video_name = sys.argv[1]

vd_name = video_name.split('/')[-1].split('.')[0]
print(vd_name)

cap = cv2.VideoCapture(video_name)
count = 0
fps = int(cap.get(5))                     #return the number of frame per second of the video
print("FPS = ", fps)

print("Mode: ", cap.get(9))
time_interval = 5
time_step = time_interval * fps           #We just keep 1 fram every time_interval

FRAMEDIR = "./Frames/"+vd_name+"_Frames/"


"""Create directory for the extraction and extracts"""

try:
    if not os.path.exists(FRAMEDIR):
        os.makedirs(FRAMEDIR)
except OSERROR:
    print('Error: Creating directory of data')
while True:
    ret, frame = cap.read()
    count += 1
    if ret == True:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if count % time_step == 0:                        #We just want to keep 1 frame every 5 seconds
            name = FRAMEDIR + vd_name + "_frame_" + str(int(count/time_step)) + '.jpg'
            print(name)
            cv2.imwrite(name,frame)
            # cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
