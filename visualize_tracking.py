# visualize_tracking.py
import cv2

def draw_tracks(video_path, tracking_output):
    cap = cv2.VideoCapture(video_path)
    for frame_idx in range(len(tracking_output)):
        ret, frame = cap.read()
        for bbox, track_id in tracking_output[frame_idx]:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, str(track_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) == 27: break
    cap.release()
    cv2.destroyAllWindows()
