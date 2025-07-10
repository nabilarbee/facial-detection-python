opencv-python==4.6.0.66
mediapipe==0.10.21

main reference: https://www.youtube.com/watch?v=DRMBqhrfxXg


elif args.mode in ['webcam']:
    cap = cv2.VideoCapture(0)

    output_video = cv2.VideoWriter(os.path.join(output_dir, 'outputw.mp4'),
                            cv2.VideoWriter_fourcc(*'MP4V'),
                            25,
                            (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_image(frame, face_detection)

        output_video.write(frame)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()