from flask import Flask, render_template, request, send_from_directory, url_for, redirect
from flask_socketio import SocketIO, emit
import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import time
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# create lock to manage access to mediapipe
latest_frame = None
lock = threading.Lock()
webcam_blur_intensity = 51 
is_recording = False
video_writer = None
recording_filename = None

# fps variables
fps_start_time = 0
fps_frame_count = 0
measured_fps = 16.4

# config outputs and allowed file extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# initialise once so it doesnt have to reload the model everytime a request is made
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
video_face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
# function to check if the file has correct extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# main function to process and blur faces in image
# reference from youtube https://www.youtube.com/watch?v=DRMBqhrfxXg
def process_and_blur_faces(img, detection_model, blur_intensity=51):
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return img
    
    # cv2.GaussianBlur requires the kernel size to be odd.
    # We ensure the received intensity is at least 1 and is odd.
    kernel_val = max(1, int(blur_intensity))
    if kernel_val % 2 == 0:
        kernel_val += 1
    blur_kernel_size = (kernel_val, kernel_val)
    # print(f"Processing with blur intensity: {kernel_val}")
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detection_model.process(img_rgb)


    if out.detections:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # calc initial bounding box coordinates
            x1, y1 = int(bbox.xmin * W), int(bbox.ymin * H)-int(bbox.height * H / 2)
            w, h = int(bbox.width * W), int(bbox.height * H)+int(bbox.height * H / 2)
            x2, y2 = x1 + w, y1 + h

            # clamp coordinates to ensure they are within the image boundaries
            x1_clamped = max(0, x1)
            y1_clamped = max(0, y1)
            x2_clamped = min(W, x2)
            y2_clamped = min(H, y2)

            # check if clamped region has valid, non-zero area.
            if (x2_clamped - x1_clamped) > 0 and (y2_clamped - y1_clamped) > 0:
                # slice image using only clamped coordinates.
                face_roi = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                
                # apply gaussian blur to safe region
                blurred_face = cv2.GaussianBlur(face_roi, blur_kernel_size, 0)
                
                # place blurred region back into image
                img[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = blurred_face
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
        
        blur_intensity_from_form = request.form.get('blur_intensity', default=51, type=int)

        # creating unique file name
        original_filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{original_filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)

        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # check file extension for either image or video
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        if file_ext in ['png', 'jpg', 'jpeg']:
            img = cv2.imread(input_path)
            processed_img = process_and_blur_faces(img, face_detection, blur_intensity_from_form)
            cv2.imwrite(output_path, processed_img)
        elif file_ext in ['mp4', 'mov', 'avi']:
            process_video_file(input_path, output_path, blur_intensity_from_form)
        
        return render_template('index.html', processed_file=output_filename)
    # path for initial preview
    initial_preview_src = ''
    preview_img_path = 'preview/preview.jpeg'
    if os.path.exists(preview_img_path):
        img = cv2.imread(preview_img_path)
        processed_img = process_and_blur_faces(img, face_detection, 51)
        if processed_img is not None:
            _, buffer = cv2.imencode('.jpg', processed_img)
            initial_preview_src = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            
    return render_template('index.html', processed_file=None, initial_preview_src=initial_preview_src)

@app.route('/outputs/<filename>')
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

# function to build on previous one so that it can process videos
def process_video_file(input_path, output_path, blur_intensity):
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
        frame = process_and_blur_faces(frame, video_face_detection, blur_intensity)
        # video writes the frame
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()

# BACKGROUND THREAD for webcam processing
def background_processing_thread():
    """
    This is the consumer thread. It continuously processes the latest frame.
    """
    global latest_frame, lock, video_writer, is_recording, webcam_blur_intensity, recording_filename
    global fps_start_time, fps_frame_count, measured_fps

    # initialise starting time
    fps_start_time = time.time()

    while True:
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                # dupe, process, release lock
                frame_to_process = latest_frame.copy()
        
        if frame_to_process is not None:
            # process outside lock to avoid thread block
            # print("Processing frame with blur intensity: ", webcam_blur_intensity)
            processed_frame = process_and_blur_faces(frame_to_process, face_detection, webcam_blur_intensity)
            
            if processed_frame is not None:
                processed_frame = cv2.flip(processed_frame, 1)

                # calculate fps when start webcam
                fps_frame_count += 1
                elapsed_time = time.time() - fps_start_time
                if elapsed_time > 1.0: # update fps calculation every second
                    measured_fps = fps_frame_count / elapsed_time
                    print(f"Actual processing FPS: {measured_fps:.2f}")
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                if is_recording:
                    # make video writer
                    if video_writer is None:
                        pframe_height, pframe_width, _ = processed_frame.shape
                        recording_filename = f"recording_{int(time.time())}.mp4"
                        output_path = os.path.join(app.config['OUTPUT_FOLDER'], recording_filename)
                        print(f"FPS for recording: {measured_fps:.2f}")
                        video_writer = cv2.VideoWriter(output_path,
                                                      cv2.VideoWriter_fourcc(*'mp4v'),
                                                      measured_fps,
                                                      (pframe_width, pframe_height))
                    # write frame to video
                    video_writer.write(processed_frame)
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('response_back', 'data:image/jpeg;base64,' + frame_data)
        
        # control processing speed
        socketio.sleep(0.03)

# webcam socketio connection
@socketio.on('connect')
def connect():
    """
    Start the background thread when a client connects.
    """
    # MAKE SURE ONLY ONE BACKGROUND THREAD RUNNING
    if not hasattr(app, 'background_thread_started'):
        socketio.start_background_task(target=background_processing_thread)
        app.background_thread_started = True
    print("Client connected")


# route for webcam
@socketio.on('image')
def image(data_image):
    # receives a frame, and update the global variable for the frame
    global latest_frame, lock
    try:
        sbuf = base64.b64decode(data_image.split(",")[1])
        npimg = np.frombuffer(sbuf, dtype=np.uint8)
        frame = cv2.imdecode(npimg, 1)

        with lock:
            latest_frame = frame
            
    except Exception as e:
        # ignore if bad frame sent
        print(f"Error processing frame: {e}")
        pass

# handler to listen for blur update from client
@socketio.on('update_blur')
def handle_blur_update(data):
    global webcam_blur_intensity
    try:
        intensity = int(data.get('intensity', 51))
        if 0 <= intensity <= 99:
            webcam_blur_intensity = intensity
            # print(f"Webcam blur intensity set to: {webcam_blur_intensity}")
    except (ValueError, TypeError):
        pass

# handler for live preview image
@socketio.on('get_preview')
def handle_preview_request(data):
    try:
        intensity = int(data.get('intensity', 51))
        print(f"Generating preview with intensity: {intensity}")
        preview_img_path = 'preview/preview.jpeg'
        if os.path.exists(preview_img_path):
            img = cv2.imread(preview_img_path)
            processed_img = process_and_blur_faces(img, face_detection, intensity)
            if processed_img is not None:
                _, buffer = cv2.imencode('.jpg', processed_img)
                preview_src = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
                # send preview image back to client
                emit('preview_updated', {'src': preview_src})
    except Exception as e:
        print(f"Error generating preview: {e}")

# handler for toggle recording
@socketio.on('toggle_recording')
def handle_toggle_recording(data):
    global is_recording, video_writer, recording_filename
    is_recording = data.get('recording', False)

    if is_recording:
        print("Recording started...")
    else:
        print("Recording stopped.")
        if video_writer is not None:
            filename_to_emit = recording_filename
            video_writer.release()
            video_writer = None
            recording_filename = None
            if filename_to_emit:
                emit('recording_complete', {'filename': filename_to_emit})
if __name__ == '__main__':
    socketio.run(app, debug=True)