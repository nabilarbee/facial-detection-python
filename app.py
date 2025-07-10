import os
import argparse
import cv2
import mediapipe as mp
import sys

def process_and_blur_faces(img, face_detection, blur_kernel_size=(75, 75)):
    """
    Detects faces in an image and applies a blur effect to them.

    Args:
        img: The input image (NumPy array).
        face_detection: The MediaPipe face detection model instance.
        blur_kernel_size: A tuple specifying the width and height of the blur kernel.

    Returns:
        The image with detected faces blurred.
    """
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Ensure the bounding box is within the image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x1 + w), min(H, y1 + h)

            # Apply blur to the face region
            if x1 < x2 and y1 < y2:
                img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], blur_kernel_size)
    
    return img

def process_image_file(face_detection, input_path, output_dir):
    """
    Loads an image, processes it to blur faces, and saves the result.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return

    processed_img = process_and_blur_faces(img, face_detection)
    
    output_path = os.path.join(output_dir, 'output.png')
    cv2.imwrite(output_path, processed_img)
    print(f"Processed image saved to {output_path}")

def process_video_file(face_detection, input_path, output_dir):
    """
    Processes a video file to blur faces in each frame and saves the result.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        cap.release()
        return
        
    frame_height, frame_width, _ = frame.shape
    output_path = os.path.join(output_dir, 'output.mp4')
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame_width, frame_height))

    while ret:
        frame = process_and_blur_faces(frame, face_detection)
        output_video.write(frame)
        ret, frame = cap.read()

    cap.release()
    output_video.release()
    print(f"Processed video saved to {output_path}")

def process_webcam(face_detection, output_dir, record=False):
    """
    Captures video from the webcam, blurs faces in real-time, and optionally records the output.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    output_video = None
    if record:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = os.path.join(output_dir, 'output_webcam.mp4')
        output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame_width, frame_height))
        print(f"Recording webcam feed to {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        frame = process_and_blur_faces(frame, face_detection)
        frame = cv2.flip(frame, 1)

        if record and output_video:
            output_video.write(frame)

        cv2.imshow('Webcam Face Blur', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_video:
        output_video.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to parse arguments and run the face blurring application.
    """
    parser = argparse.ArgumentParser(description="Blur faces in images, videos, or webcam streams.")
    parser.add_argument("--mode", default='webcam', choices=['image', 'video', 'webcam'], 
                        help="The processing mode: 'image', 'video', or 'webcam'.")
    parser.add_argument("--filePath", default=None, 
                        help="Path to the input image or video file (required for 'image' and 'video' modes).")
    parser.add_argument("--record", action='store_true', 
                        help="Enable recording when using 'webcam' mode.")
    
    args = parser.parse_args()

    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        if args.mode == "image":
            if not args.filePath:
                sys.exit("Error: --filePath is required for 'image' mode.")
            process_image_file(face_detection, args.filePath, output_dir)
        elif args.mode == 'video':
            if not args.filePath:
                sys.exit("Error: --filePath is required for 'video' mode.")
            process_video_file(face_detection, args.filePath, output_dir)
        elif args.mode == 'webcam':
            process_webcam(face_detection, output_dir, args.record)

if __name__ == '__main__':
    main()