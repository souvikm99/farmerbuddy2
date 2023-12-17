import cv2
import numpy as np
import math
from ultralytics import YOLO

cap = cv2.VideoCapture("randomtomato.mp4")

model = YOLO("best.pt")
classNames = ["green-chilli", "redchilliS", "soya", "tomato"]
frame_count = 0
center_points_prev_frame = []
tracking_objects = {}
track_id = 0

total_obj_dict = {}

frame_change = 40

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break

    #center points current frame
    center_points_current_frame = []

    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            center_points_current_frame.append((cx,cy))

            conf = math.ceil((box.conf[0] * 100)) / 100
            class_id = int(box.cls[0])
            class_name = classNames[class_id]

            print(class_id, class_name, conf)

            # cv2.circle(frame,(cx,cy),5,(255,255,255),-1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #only at the begining we compare prev and cur frame

    if frame_count <= 2:
        for pt in center_points_current_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])

                if distance < frame_change:
                    tracking_objects[track_id] = pt
                    track_id += 1

    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_current_frame_copy = center_points_current_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_current_frame_copy:
                distance = math.hypot(pt2[0]-pt[0],pt2[1]-pt[1])

                #update ids positions
                if distance < frame_change :
                    tracking_objects[object_id] = pt
                    object_exists = True

                    if pt in center_points_current_frame:
                        center_points_current_frame.remove(pt)
                    continue
            
            #remove id lost
            if not object_exists:
                tracking_objects.pop(object_id)


        #add new ids found
        for pt in center_points_current_frame:

            tracking_objects[track_id] = pt
            track_id += 1

    for object_id,pt in tracking_objects.items():
        total_obj_dict[object_id] = class_name
        cv2.circle(frame,pt,5,(255,255,255),-1)
        cv2.putText(frame,str(f"{object_id}-{class_name}({conf*100:.2f}%)"),(pt[0],pt[1]-7),0,1,(0,0,255),1)
                
        # cv2.circle(frame,pt,5,(255,255,255),-1)

    cv2.imshow("Frame", frame)

    #make a copy of current center points
    center_points_prev_frame = center_points_current_frame.copy()
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

#printing total object containig dict, where id:class
print(total_obj_dict)
