import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker

#video_path= os.path.join('.','data','people.mp4')
video_path= os.path.join('.','data','video.mp4')
video_out_path= os.path.join('.','out.mp4')

cap= cv2.VideoCapture(video_path)
ret, frame= cap.read()

#Save de video we need specify the location and other 3 inputs, one 
# is related to the codec. The second specify the frames per second 
# in this video, this output video we want to make it the same frames 
# per second as the original video.
cap_out= cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                        (frame.shape[1], frame.shape[0])) #we specify the size

model= YOLO('yolov8n.pt') #Pre-trained model to Detection 'https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models'

tracker= Tracker() #We need initialize it, I'm going to create a instance

#Creating now a variable list of colors which contains 10 completely random colors
colors= [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(10)] #For the bbox

detection_threshold= 0.5  #To have good result
while ret:

    results= model(frame)
    
    for result in results:
        detections= []
        for r in result.boxes.data.tolist():
            print(r) # One result is [564.0, 35.0, 592.0, 122.0, 0.6188790798187256, 0.0]
            x1, y1, x2, y2, score, class_id= r #The class id referent to coco_classes.txt
            class_id= int(class_id)
            if score > detection_threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), score])

        #Here we going to update all the tracking information because for the way object 
        # tracking works, for the way deep sort works, the algorithm compute some features 
        # on top of the frame on top of the objects which were detected. It's going to crop 
        # the frame and it's going to extract some features from it, so that we 
        # need we definitely need to input the frame and then detections
        tracker.update(frame, detections)

        #Now we need to iterate in all the tracks in this algorithm
        for track in tracker.tracks:
            bbox= track.bbox
            x1, y1, x2, y2= bbox

            #ID is the most impotant information from the object tracking algorithm which 
            # is going to be the ID the tracking algorithm has assigned to this object. 
            #The ID that object tracking agree in deep sort has assigned to this object, 
            # that will be exactly the same across all the frames in this video. In this 
            # case, we are going to detect are people and each one of these persons is 
            # going to have an ID assigned so the object tracking algorithm deep sort is 
            # going to assing a given number (ID) to each person and it's going to keep the 
            # same ID across all the video.
            track_id= track.track_id

            #Now plotting a bonding box, a rectangle on top of the frame for each one 
            # of these objects. Every time we get a new identifier we are going to 
            # plot this new ID, this new object we are detecting with a new color.
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)


    #cv2.imshow('Frame', frame)
    #cv2.waitKey(25)

    cap_out.write(frame)  #Write the frames of the output video
    ret, frame= cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()