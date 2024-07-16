#Code with comments
import cv2 as cv
import mediapipe as mp
import numpy as np

mpfacemesh = mp.solutions.face_mesh
FaceMesh = mpfacemesh.FaceMesh(max_num_faces=1)
mpdraw = mp.solutions.drawing_utils
drawspec1 = mpdraw.DrawingSpec(color = (255,255,0), circle_radius = 0, thickness = 1)
drawspec2 = mpdraw.DrawingSpec(color = (0,255,0), circle_radius = 0, thickness = 1)
webcam = cv.VideoCapture(0)

#following indices are available in mediapipe dev site
EYE_LEFT_CONTOUR = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
EYE_RIGHT_CONTOUR = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYEBROW = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
RIGHT_EYEBROW = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
while True:
  
 scc,img = webcam.read()
 img = cv.flip(img,1)
 h,w,c = img.shape
 results = FaceMesh.process(img)
 
 if results.multi_face_landmarks:
  for face_lm in results.multi_face_landmarks:
   X=[]
   Y=[]
   for lm in face_lm.landmark:
    X.append(int(lm.x*w))
    Y.append(int(lm.y*h))
   #left eye center
   xl = int(np.mean([X[i] for i in EYE_LEFT_CONTOUR]))
   yl = int(np.mean([Y[i] for i in EYE_LEFT_CONTOUR]))
   cv.circle(img,(xl,yl),9,(255,0,255),7)
   #right eye center
   xr = int(np.mean([X[i] for i in EYE_RIGHT_CONTOUR]))
   yr = int(np.mean([Y[i] for i in EYE_RIGHT_CONTOUR]))
   cv.circle(img,(xr,yr),9,(255,0,255),7)
   cv.line(img,(xl,yl),(xr,yr),(255,0,255),3)
   #eyebrows
   xlb = int(np.mean([X[i] for i in LEFT_EYEBROW]))
   ylb = int(np.mean([Y[i] for i in LEFT_EYEBROW]))
   xrb = int(np.mean([X[i] for i in RIGHT_EYEBROW]))
   yrb = int(np.mean([Y[i] for i in RIGHT_EYEBROW]))
   #final drawing
   cv.putText(img,'*',(xl-9,yl+9),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
   cv.putText(img,'*',(xr-9,yr+9),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
   cv.putText(img,'^',(xlb-9,ylb),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
   cv.putText(img,'^',(xrb-9,yrb),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
   
 k = cv.waitKey(1)
 if k == ord('q'):
  break
 cv.imshow('augmented reality', img)
webcam.release()  
cv.destroyAllWindows()