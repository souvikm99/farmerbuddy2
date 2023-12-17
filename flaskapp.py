from flask import Flask, render_template, Response, request, session, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2
import math
import datetime
from ultralytics import YOLO
from deepSort0.helper import create_video_writer
from deepSort0.deep_sort_realtime.deepsort_tracker import DeepSort
import base64
import numpy as np
from openpyxl import Workbook
import csv

app = Flask(__name__)

app.config['SECRET_KEY'] = 'souvik'
app.config['UPLOAD_FOLDER'] = 'static/files/'

lol2 = ""

# Define a global variable to store the processed video path
processed_video_path = ""


def generate_frames(path_x=''):
    global processed_video_path

    yolo_output = video_detection(path_x)

    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release video writer after processing
    processed_video_path = None

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    text1 = count1(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        lol2 = text1
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print(">>>>>", lol2)


def video_detection(path_x):
    global processed_video_path

    video_capture = path_x

    CONFIDENCE_THRESHOLD = 0.8
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    total_obj_dict = {}

    classNames = [
        "avocado", "beans", "beet", "bell pepper", "broccoli", "brus capusta", "cabbage", "carrot", "cauliflower",
        "celery", "corn", "cucumber", "eggplant", "fasol", "garlic", "hot pepper", "onion", "peas", "potato",
        "pumpkin", "rediska", "redka", "salad", "squash-patisson", "tomato", "vegetable marrow"
    ]

    video_cap = cv2.VideoCapture(video_capture)
    writer = create_video_writer(video_cap, "output.mp4")

    model = YOLO("/Users/souvikmallick/Desktop/YoloFlask_2-master/best_v8_robo_data.pt")
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

        writer.write(frame)

        yield frame

    writer.release()
    processed_video_path = "output.mp4"


def count1(path_x):
    video_capture = path_x

    CONFIDENCE_THRESHOLD = 0.8
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    total_obj_dict = {}
    classNames = [
        "avocado", "beans", "beet", "bell pepper", "broccoli", "brus capusta", "cabbage", "carrot", "cauliflower",
        "celery", "corn", "cucumber", "eggplant", "fasol", "garlic", "hot pepper", "onion", "peas", "potato",
        "pumpkin", "rediska", "redka", "salad", "squash-patisson", "tomato", "vegetable marrow"
    ]

    video_cap = cv2.VideoCapture(video_capture)
    writer = create_video_writer(video_cap, "output.mp4")

    model = YOLO("/Users/souvikmallick/Desktop/YoloFlask_2-master/best_v8_robo_data.pt")
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

        yield lol


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')


@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.files.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form, var2=lol2)


@app.route('/video')
def video():
    global detected_frames
    detected_frames = []  # Reset detected frames
    return Response(generate_frames(path_x=session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/processed_video')
def processed_video():
    global processed_video_path
    if processed_video_path:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
        return render_template('processed_video.html', video_path=video_path)
    else:
        return "Video processing is not complete yet. Please wait."


@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')


class UploadFileForm(FlaskForm):
    files = FileField("Upload Images", validators=[InputRequired()])
    submit = SubmitField("Detect Objects")


def process_uploaded_images(files):
    CONFIDENCE_THRESHOLD = 0.8
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    classNames = [
        "avocado", "beans", "beet", "bell pepper", "broccoli", "brus capusta", "cabbage", "carrot", "cauliflower",
        "celery", "corn", "cucumber", "eggplant", "fasol", "garlic", "hot pepper", "onion", "peas", "potato",
        "pumpkin", "rediska", "redka", "salad", "squash-patisson", "tomato", "vegetable marrow"
    ]

    model = YOLO("/Users/souvikmallick/Desktop/YoloFlask_2-master/best_v8_robo_data.pt")
    tracker = DeepSort(max_age=50)

    object_counts = {}  # Initialize as an empty dictionary
    detected_images = []  # Initialize an empty list to store detected images

    for file in files:
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        detections = model(image)[0]

        for data in detections.boxes.data.tolist():
            confidence = data[4]

            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(data[5])
            class_name = classNames[class_id]

            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1

            # Draw bounding boxes on the image for counted objects
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(image, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
            cv2.putText(image, f"{class_name}", (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Encode the detected image as a base64 string and append it to the list
        _, buffer = cv2.imencode('.jpg', image)
        detected_images.append(base64.b64encode(buffer).decode())

    return detected_images, object_counts


@app.route('/image', methods=['GET', 'POST'])
def image():
    form = UploadFileForm()
    detected_images = None
    object_counts = None

    if form.validate_on_submit():
        files = request.files.getlist('files')  # Get a list of uploaded files
        detected_images, object_counts = process_uploaded_images(files)

    if object_counts is None:
        object_counts = {}  # Initialize as an empty dictionary

    return render_template('image_upload.html', form=form, detected_images=detected_images, object_counts=object_counts)


@app.route('/export_csv', methods=['POST'])
def export_to_csv():
    class_names = request.form.getlist('class_name[]')
    class_counts = request.form.getlist('class_count[]')

    # Create a list to store CSV data
    csv_data = [["Class Name", "Count"]]

    # Add data to the CSV list
    for class_name, count in zip(class_names, class_counts):
        csv_data.append([class_name.strip(), count.strip()])

    # Create a temporary CSV file
    csv_filename = 'object_counts.csv'
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_data)

    # Send the CSV file for download
    return send_file(csv_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
