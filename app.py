from flask import Flask, render_template, request, send_from_directory, url_for, redirect
from flask_socketio import SocketIO, emit
import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
socketio = SocketIO(app)

# config outputs and allowed file extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialise once so it doesnt have to reload the model everytime a request is made
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# function to check if the file has correct extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# main function to process and blur faces in image
# reference from youtube https://www.youtube.com/watch?v=DRMBqhrfxXg
def process_and_blur_faces(img, blur_kernel_size=(75, 75)):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * W), int(bbox.ymin * H)
            w, h = int(bbox.width * W), int(bbox.height * H)
            img[y1:y1+h, x1:x1+w, :] = cv2.GaussianBlur(img[y1:y1+h, x1:x1+w, :], blur_kernel_size, 0)
    return img

# routes for file processing buttons
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # checking if it's a file upload
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        # if no file selected or file not allowed
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        # creating unique file name
        original_filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{original_filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)

        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Checking file extension for either image or video
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        if file_ext in ['png', 'jpg', 'jpeg']:
            img = cv2.imread(input_path)
            processed_img = process_and_blur_faces(img)
            cv2.imwrite(output_path, processed_img)
        elif file_ext in ['mp4', 'mov', 'avi']:
            process_video_file(input_path, output_path)
        
        return render_template('index.html', processed_file=output_filename)

    return render_template('index.html', processed_file=None)

@app.route('/outputs/<filename>')
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

# function to build on previous one so that it can process videos
def process_video_file(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    
    frame_height, frame_width, _ = frame.shape
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width, frame_height))

    # while there are still frames to be processed
    while ret:
        # current frame goes to the function built earlier to process
        frame = process_and_blur_faces(frame)
        # video writes the frame
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()

# Socketio route for webcam
@socketio.on('image')
def image(data_image):
    # decode image from base64
    sbuf = base64.b64decode(data_image.split(",")[1])
    npimg = np.frombuffer(sbuf, dtype=np.uint8)
    frame = cv2.imdecode(npimg, 1)

    # process frame to blur face
    frame = process_and_blur_faces(frame)
    
    # flip webcam
    frame = cv2.flip(frame, 1)
    
    # change back to jpg file and base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    
    # send frame to html
    emit('response_back', 'data:image/jpeg;base64,' + frame_data)

if __name__ == '__main__':
    socketio.run(app, debug=True)