"""
HOME VERSION
threshold_counturs.py: Thresholds an image and finds contours and detects balls for FRC 2022
    inspired from: https://github.com/rebels2638/frc-vision-2022/blob/main/countour_threshold_method/threshold_contours.py, https://stackoverflow.com/questions/58016325/thinness-ratio-calculation-wrong-results?noredirect=1&lq=1
"""

__author__ = "Caden Li"
__copyright__ = "Copyright 2022, Caden Li"
__credits__ = ["Collin Li"]
__license__ = "MIT"


from flask import Flask, render_template, Response
import numpy as np
import imutils
import cv2
import math
import time



lower = {'red':(0, 106, 199),'blue':(39, 114, 152)} 
upper = {'red':(179,255,255),'blue':(101,255,255)}
 

colors = {'red':(0,0,255), 'blue':(255,0,0)}

camera = cv2.VideoCapture(0)
 
app = Flask(__name__,template_folder='templates')

avg = 0
count = 0

def ballDetection():
    while True:
    
        (grabbed, frame) = camera.read()
        
        starttime = time.time()
      
        #frame = imutils.resize(frame, width=900)
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
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
            #center = None
           
    
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
                if radius > 5 and ti > .5:
                
                    cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 5)
                    cv2.circle(frame, (int(x), int(y)), 1, colors[key], 5)
                    cv2.putText(frame,sti, (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                    
     
        fps_int = round(1.0/(time.time()-starttime),2)
        fps = "FPS (circles + regular): "+str(fps_int)
        cv2.putText(frame,fps,(60,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0))[2]
    
        retval, buff_detect = cv2.imencode(".jpg",frame) # turns evrything into jpg
        frame_detect = buff_detect.tobytes() # converts the buffer to bytes
    
        yield (b'--frame_detect\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_detect + b'\r\n') # yield output
 


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/videoBallDetection')

def videoBallDetection():
    return Response(ballDetection(),mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)