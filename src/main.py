import cv2
from eye_tracker import EyeTracker

def main(video_source=0):
    # Initialize the eye tracker
    tracker = EyeTracker()

    # Open the video source
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        tracker.process_frame(frame)

        # Display the frame
        cv2.imshow('Eye Tracking', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    main(video_source) 