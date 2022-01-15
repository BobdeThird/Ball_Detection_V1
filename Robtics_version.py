"""
ROBOTICS VERSION
threshold_counturs.py: Thresholds an image and finds contours and detects balls for FRC 2022
    inspired from: https://github.com/rebels2638/frc-vision-2022/blob/main/countour_threshold_method/threshold_contours.py, https://stackoverflow.com/questions/58016325/thinness-ratio-calculation-wrong-results?noredirect=1&lq=1
"""

__author__ = "Caden Li"
__copyright__ = "Copyright 2022, Caden Li"
__credits__ = ["Collin Li"]
__license__ = "MIT"

from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import urllib 
import math
import time


lower = {'red':(0, 133, 110),'blue':(97, 100, 117)} 
upper = {'red':(179,255,255),'blue':(117,255,255)}
 

colors = {'red':(0,0,255), 'blue':(255,0,0)}
 


camera = cv2.VideoCapture(0)


while True:

    (grabbed, frame) = camera.read()
  
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    area = 0
    perimeter = 1
    
    for key, value in upper.items():
    
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
               
       
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
       

        #if len(cnts) > 0:
        for i in cnts:
            
            c = i #max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(i)
            perimeter = cv2.arcLength(i, True)
            
            #M = cv2.moments(c)
            #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
       
            ti = (4*area*math.pi)/(perimeter**2) 
            ti = round(ti, 2)
            sti = str(ti)
            if radius > 20 and ti > .5:
            
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 5)
                cv2.circle(frame, (int(x), int(y)), 1, colors[key], 5)
                cv2.putText(frame,sti, (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                
 

    ti = (4*area*math.pi)/(perimeter**2) 
    #print(ti)
    cv2.imshow("Frame", frame)
   
    key = cv2.waitKey(1) & 0xFF
    # press 'q' to stop the loop
    if key == ord("q"):
        break
    
    

 
camera.release()
cv2.destroyAllWindows()

