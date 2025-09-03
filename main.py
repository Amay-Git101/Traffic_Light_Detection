import cv2
import numpy as np

# ---------- SETTINGS ----------
# Use 0 for webcam, or provide a video path like "videos/traffic.mp4"
VIDEO_SOURCE = 0

# It's crucial to handle the two ranges for Red's hue, as it wraps around 0/180
# These are optimized ranges that should work well in most conditions.
OPTIMIZED_HSV_RANGES = {
    # (lower_hsv_range, upper_hsv_range)
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Yellow": [
        (np.array([20, 100, 100]), np.array([30, 255, 255]))
    ],
    "Green": [
        (np.array([40, 70, 70]), np.array([90, 255, 255]))
    ]
}

# Minimum contour area to filter out small noise
MIN_CONTOUR_AREA = 500

# ---------- VIDEO CAPTURE ----------
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error.")
        break

    # 1. CONVERT FRAME FROM BGR TO HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_color = "None"
    max_area = 0
    best_contour_info = None # Will store (x, y, w, h) for the best contour

    # Loop through each color configuration
    for color_name, hsv_ranges in OPTIMIZED_HSV_RANGES.items():
        # Create a combined mask for the current color
        # This is especially important for red, which has two ranges
        current_mask = None
        for hsv_range in hsv_ranges:
            lower_bound, upper_bound = hsv_range
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            if current_mask is None:
                current_mask = mask
            else:
                current_mask = cv2.add(current_mask, mask)

        # 2. APPLY MORPHOLOGICAL OPERATIONS
        # This helps to remove small noise from the mask
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # 3. DETECT CONTOURS
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 4. FIND THE LARGEST CONTOUR FOR THE CURRENT COLOR
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area
            if area > MIN_CONTOUR_AREA:
                # We assume the largest detected blob is the active light
                if area > max_area:
                    max_area = area
                    detected_color = color_name
                    best_contour_info = cv2.boundingRect(contour)

    # 5. DRAW BOUNDING BOX AND ADD TEXT ON THE ORIGINAL FRAME
    if detected_color != "None" and best_contour_info is not None:
        x, y, w, h = best_contour_info
        
        # Define color for the bounding box based on the detected light
        if detected_color == "Red":
            box_color = (0, 0, 255)
        elif detected_color == "Yellow":
            box_color = (0, 255, 255)
        else: # Green
            box_color = (0, 255, 0)
            
        # Draw the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        # Put the text label
        cv2.putText(frame, f"Active Light: {detected_color}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Display the final processed frame
    cv2.imshow("Traffic Light Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

def test_on_image(image_path):
    """
    Function to run the detection logic on a single static image and display it.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # The same detection logic from the main loop is used here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_color = "None"
    max_area = 0
    best_contour_info = None

    for color_name, hsv_ranges in OPTIMIZED_HSV_RANGES.items():
        current_mask = None
        for hsv_range in hsv_ranges:
            lower_bound, upper_bound = hsv_range
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            if current_mask is None:
                current_mask = mask
            else:
                current_mask = cv2.add(current_mask, mask)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                if area > max_area:
                    max_area = area
                    detected_color = color_name
                    best_contour_info = cv2.boundingRect(contour)

    if detected_color != "None" and best_contour_info is not None:
        x, y, w, h = best_contour_info
        box_color = (0, 0, 255) if detected_color == "Red" else (0, 255, 255) if detected_color == "Yellow" else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame, f"Active Light: {detected_color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Show the result in a window
    cv2.imshow(f"Static Image Test - Detected: {detected_color}", frame)
    print(f"Detected Color: {detected_color}")
    cv2.waitKey(0) # Wait until a key is pressed
    cv2.destroyAllWindows()

# --- To run the static image test ---
# 1. Comment out the main "while True:" loop for the webcam feed.
# 2. Uncomment the line below and provide the path to your test image.
# test_on_image('path/to/your/traffic_light.jpg')