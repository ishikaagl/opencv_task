import math
import cv2
import numpy as np

# ---------------- Settings ----------------
CAMERA_NUMBER = 0
WINDOW_NAME = "Blue Object and Tennis Ball Detection"

# HSV colour ranges
BLUE_LOWER = np.array([90, 70, 40])
BLUE_UPPER = np.array([135, 255, 255])
BALL_LOWER = np.array([25, 120, 100])    # vivid yellow-green (high saturation rejects beige walls)
BALL_UPPER = np.array([45, 255, 255])

# Detection thresholds
MIN_BLUE_AREA = 500          # px^2
MIN_BALL_AREA = 400          # px^2
MIN_BALL_CIRCULARITY = 0.70  # 1.0 = perfect circle

# Tracking
MAX_TRACK_DISTANCE = 60      # max px a centre may move between frames
MAX_MISSING_FRAMES = 10      # drop a track after this many frames unseen
MAX_PATH_LENGTH = 25         # number of past centres kept for the trail

# Drawing colours (BGR)
BLUE_COLOR = (255, 100, 0)
BALL_COLOR = (0, 255, 0)
LINE_COLOR = (0, 255, 255)
INFO_COLOR = (0, 200, 200)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
KERNEL = np.ones((5, 5), np.uint8)


# ---------------- Helpers ----------------
def put_text(frame, text, position, color, scale=0.5):
    """Draw text on the frame with the default font and thickness."""
    cv2.putText(frame, text, position, FONT, scale, color, 2)


def clean_mask(mask):
    """Remove small noise and fill small gaps in a colour mask."""
    mask = cv2.erode(mask, KERNEL, iterations=1)
    return cv2.dilate(mask, KERNEL, iterations=2)


def find_contours(mask):
    """Find outer contours (works with both OpenCV return formats)."""
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


# ---------------- Blue objects ----------------
def detect_blue_objects(hsv):
    """Return a list of blue objects, each with area, box and centre."""
    mask = clean_mask(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER))
    objects = []

    for contour in find_contours(mask):
        area = cv2.contourArea(contour)
        if area < MIN_BLUE_AREA:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        objects.append({
            "area": area,
            "box": cv2.boundingRect(contour),
            "center": (int(moments["m10"] / moments["m00"]),
                       int(moments["m01"] / moments["m00"])),
        })
    return objects


def update_tracks(tracks, blue_objects, frame_number, next_id):
    """
    Match each blue object to the nearest recent track (or start a new one),
    store its trail, and delete tracks that have disappeared.
    Returns the next unused track id.
    """
    free_ids = set(tracks)   # old tracks not yet matched this frame

    for obj in blue_objects:
        cx, cy = obj["center"]
        best_id, best_distance = None, MAX_TRACK_DISTANCE

        for track_id in free_ids:
            old_x, old_y = tracks[track_id]["center"]
            distance = math.hypot(cx - old_x, cy - old_y)
            if distance < best_distance:
                best_id, best_distance = track_id, distance

        if best_id is None:                       # no close track -> new object
            best_id = next_id
            next_id += 1
            tracks[best_id] = {"center": (cx, cy), "path": [], "last_seen": frame_number}

        free_ids.discard(best_id)
        track = tracks[best_id]
        track["center"] = (cx, cy)
        track["last_seen"] = frame_number
        track["path"] = (track["path"] + [(cx, cy)])[-MAX_PATH_LENGTH:]
        obj["track_id"] = best_id

    for track_id in list(tracks):
        if frame_number - tracks[track_id]["last_seen"] > MAX_MISSING_FRAMES:
            del tracks[track_id]

    return next_id


# ---------------- Tennis ball ----------------
def detect_tennis_ball(hsv):
    """Return the largest round yellow-green blob as a dict, or None."""
    mask = clean_mask(cv2.inRange(hsv, BALL_LOWER, BALL_UPPER))
    cv2.imshow("Ball mask (debug)", mask)   # white = pixels accepted as ball colour
    ball = None
    best_area = 0

    for contour in find_contours(mask):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < MIN_BALL_AREA or perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity > MIN_BALL_CIRCULARITY and area > best_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            best_area = area
            ball = {"area": area, "center": (int(x), int(y)), "radius": int(radius)}
    return ball


# ---------------- Drawing ----------------
def draw_blue_object(frame, obj, path, ball):
    x, y, w, h = obj["box"]
    cx, cy = obj["center"]

    cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE_COLOR, 2)
    cv2.circle(frame, (cx, cy), 4, BLUE_COLOR, -1)
    put_text(frame, f"Blue #{obj['track_id']}", (x, max(20, y - 35)), BLUE_COLOR, 0.6)
    put_text(frame, f"Area: {obj['area']:.0f} px^2", (x, max(40, y - 12)), BLUE_COLOR)
    put_text(frame, f"Box: {w} x {h} px", (x, y + h + 18), BLUE_COLOR)

    # Trail of past centres
    for start, end in zip(path, path[1:]):
        cv2.line(frame, start, end, BLUE_COLOR, 2)

    # Position relative to the ball
    if ball is not None:
        bx, by = ball["center"]
        cv2.line(frame, (cx, cy), (bx, by), LINE_COLOR, 2)
        put_text(frame, f"To ball: dx={bx - cx}px, dy={by - cy}px",
                 (x, min(frame.shape[0] - 10, y + h + 38)), INFO_COLOR)


def draw_tennis_ball(frame, ball):
    if ball is None:
        put_text(frame, "Tennis ball: not detected", (10, 30), RED, 0.65)
        return

    bx, by = ball["center"]
    r = ball["radius"]

    cv2.circle(frame, (bx, by), r, BALL_COLOR, 2)
    cv2.circle(frame, (bx, by), 4, RED, -1)
    put_text(frame, "Tennis Ball", (bx - r, max(20, by - r - 35)), BALL_COLOR, 0.6)
    put_text(frame, f"Center: ({bx}, {by}) px", (bx - r, max(40, by - r - 14)), BALL_COLOR)
    put_text(frame, f"Radius: {r} px  Area: {ball['area']:.0f} px^2",
             (bx - r, by + r + 20), BALL_COLOR)


# ---------------- Main ----------------
def main():
    camera = cv2.VideoCapture(CAMERA_NUMBER)
    if not camera.isOpened():
        print("Error: Could not open camera. Check CAMERA_NUMBER or camera permissions.")
        return

    tracks = {}
    next_id = 1
    frame_number = 0

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read a frame from the camera.")
            break

        frame_number += 1
        frame = cv2.flip(frame, 1)                      # mirror view
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        blue_objects = detect_blue_objects(hsv)
        next_id = update_tracks(tracks, blue_objects, frame_number, next_id)
        ball = detect_tennis_ball(hsv)

        for obj in blue_objects:
            draw_blue_object(frame, obj, tracks[obj["track_id"]]["path"], ball)
        draw_tennis_ball(frame, ball)
        put_text(frame, "Press q to quit", (10, frame.shape[0] - 15), WHITE, 0.55)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()