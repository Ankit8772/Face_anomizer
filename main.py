import os
import argparse
import cv2
import mediapipe as mp

def process_img(img, face_detection):
    if img is None:
        return None  # Avoid processing empty frames

    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)

            # Dynamic blur size based on face width
            blur_size = max(w // 5, 10)  # Min blur kernel = 10
            img[y1:y1 + h, x1:x1 + w] = cv2.blur(img[y1:y1 + h, x1:x1 + w], (blur_size, blur_size))

    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam", choices=["image", "video", "webcam"])
args.add_argument("--filePath", default=None)
args = args.parse_args()

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":
        if args.filePath is None or not os.path.exists(args.filePath):
            raise ValueError("Invalid or missing --filePath argument for image mode")

        img = cv2.imread(args.filePath)
        processed_img = process_img(img, face_detection)

        if processed_img is not None:
            cv2.imwrite(os.path.join(output_dir, "output.png"), processed_img)

    elif args.mode == "video":
        if args.filePath is None or not os.path.exists(args.filePath):
            raise ValueError("Invalid or missing --filePath argument for video mode")

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        if not ret:
            raise ValueError("Could not read video file")

        output_video = cv2.VideoWriter(
            os.path.join(output_dir, "output.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"),
            int(cap.get(cv2.CAP_PROP_FPS)),
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            processed_frame = process_img(frame, face_detection)
            if processed_frame is not None:
                output_video.write(processed_frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)  # Default webcam
        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = process_img(frame, face_detection)
            if frame is not None:
                cv2.imshow("Face Anonymizer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Press 'q' to exit

        cap.release()
        cv2.destroyAllWindows()
