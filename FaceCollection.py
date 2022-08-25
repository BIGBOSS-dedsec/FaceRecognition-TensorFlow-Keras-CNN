# Coding BIGBOSSyifi
# Datatime:2022/5/30 23:42
# Filename:FaceCollection.py
# Toolby: PyCharm

import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('data','images')
number_images = 70

cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()