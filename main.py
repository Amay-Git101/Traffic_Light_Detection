import cv2
import numpy as np
import math # Needed for circularity calculation

# ---------- SETTINGS AND CONSTANTS ----------
VIDEO_SOURCE = 0 # Use 0 for webcam, or provide a video path

# HSV Color Ranges - Tuned for better accuracy
OPTIMIZED_HSV_RANGES = {
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

# --- Detection Thresholds ---
MIN_CONTOUR_AREA = 300   # Reduced slightly for smaller lights
MIN_CIRCULARITY = 0.6    # Circularity threshold (1.0 is a perfect circle)

# --- State Machine Settings ---
STATE_CONFIRMATION_FRAMES = 5 # How many frames a state must be consistent to be confirmed

def process_frame(frame):
    """
    Processes a single video frame to detect traffic lights.
    """
    processed_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    detected_lights = [] # To store all valid lights found in the frame

    for color_name, hsv_ranges in OPTIMIZED_HSV_RANGES.items():
        current_mask = None
        for lower_bound, upper_bound in hsv_ranges:
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
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = (4 * math.pi * area) / (perimeter ** 2)
                
                if circularity > MIN_CIRCULARITY:
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_lights.append({
                        "color": color_name,
                        "area": area,
                        "box": (x, y, w, h)
                    })

    for light in detected_lights:
        x, y, w, h = light["box"]
        color_name = light["color"]
        
        box_color = (0, 0, 255) if color_name == "Red" else \
                    (0, 255, 255) if color_name == "Yellow" else (0, 255, 0)
        
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(processed_frame, color_name, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    
    return processed_frame, detected_lights

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    stable_state = "None"
    last_detected_state = "None"
    state_persistence_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, detected_lights = process_frame(frame)
        
        current_frame_state = "None"
        if detected_lights:
            largest_light = max(detected_lights, key=lambda x: x['area'])
            current_frame_state = largest_light['color']

        if current_frame_state == last_detected_state:
            state_persistence_counter += 1
        else:
            last_detected_state = current_frame_state
            state_persistence_counter = 1

        if state_persistence_counter >= STATE_CONFIRMATION_FRAMES:
            stable_state = last_detected_state

        # --- Display Logic ---
        # Define color for the text labels
        state_color = (0, 0, 255) if stable_state == "Red" else \
                      (0, 255, 255) if stable_state == "Yellow" else \
                      (0, 255, 0) if stable_state == "Green" else (255, 255, 255)
        
        # NEW LINE: Display the RAW (instantaneous) detection
        cv2.putText(processed_frame, f"RAW: {current_frame_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the STABLE (confirmed) detection
        cv2.putText(processed_frame, f"STABLE STATE: {stable_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2, cv2.LINE_AA)
        
        cv2.imshow("Traffic Light Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()