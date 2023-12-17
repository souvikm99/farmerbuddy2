import math
import datetime
from ultralytics import YOLO
import cv2
from deepSort0.helper import create_video_writer
from deepSort0.deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, render_template, Response,jsonify,request,session

path = "deepSort0/randomtomato.mp4"

def video_detection(path_x):
    video_capture = path_x

    CONFIDENCE_THRESHOLD = 0.8
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    total_obj_dict = {}
    classNames = ["green-chilli", "redchilliS", "soya", "tomato"]

    video_cap = cv2.VideoCapture(video_capture)
    writer = create_video_writer(video_cap, "output.mp4")

    model = YOLO("best.pt")
    tracker = DeepSort(max_age=50)

    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break

        detections = model(frame)[0]

        results = []

        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            class_name = classNames[class_id]
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            total_obj_dict[track_id] = class_name
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, f"{track_id}-{class_name} ({confidence*100:.2f}%)", (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"

        value_counts = {}
        lol = ""

        for value in total_obj_dict.values():
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

        for value, count in value_counts.items():
            print(f"{value}: {count}")
            lol = str(f"{value}: {count}")

        cv2.putText(frame, f"fps : {fps} | {lol}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # print("FRAME>>>", frame)

        # res = (frame, lol)
        # print("YOOOOO>>>",res[1])
        

        yield frame
        # yield lol

def count1(path_x):
    video_capture = path_x

    CONFIDENCE_THRESHOLD = 0.8
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    total_obj_dict = {}
    classNames = ["green-chilli", "redchilliS", "soya", "tomato"]

    video_cap = cv2.VideoCapture(video_capture)
    writer = create_video_writer(video_cap, "output.mp4")

    model = YOLO("best.pt")
    tracker = DeepSort(max_age=50)

    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break

        detections = model(frame)[0]

        results = []

        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            class_name = classNames[class_id]
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            total_obj_dict[track_id] = class_name
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, f"{track_id}-{class_name} ({confidence*100:.2f}%)", (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"

        value_counts = {}
        lol = ""

        for value in total_obj_dict.values():
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

        for value, count in value_counts.items():
            print(f"{value}: {count}")
            lol = str(f"{value}: {count}")

        cv2.putText(frame, f"fps : {fps} | {lol}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # print("FRAME>>>", frame)

        # res = (frame, lol)
        print("YOOOOO>>>",lol)
        

        yield lol
        # yield lol



cv2.destroyAllWindows()
