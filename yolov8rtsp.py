import json
import cv2
from datetime import datetime
from ultralytics import YOLO
import os
import sys

# Initialize the YOLO model
model = YOLO("yolov8x.pt")  # Use YOLOv8x for better performance
model.info()  # Display architecture details and number of parameters

# RTSP URL for the livestream
# Check if RTSP URL is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <rtsp_url>")
    sys.exit(1)

# Get RTSP URL from command-line argument
rtsp_url = sys.argv[1]

rtsp_cap = cv2.VideoCapture(rtsp_url)
sample_rate = 100  # Sample every 100 frames

if not rtsp_cap.isOpened():
    print(f"Error opening {rtsp_url} stream.")

frame_count = 0

# Create directory for JSON files if it doesn't exist
os.makedirs("_json", exist_ok=True)

while rtsp_cap.isOpened():
    ret, rtmp_frame = rtsp_cap.read()
    
    if not ret:
        print(f"Error reading frame {frame_count}")
        break

    # Process every 100 frames
    if frame_count % sample_rate == 0 and frame_count != 0:
        # Resize the frame while maintaining aspect ratio
        max_size = 640
        height, width = rtmp_frame.shape[:2]
        scale = max_size / max(height, width)
        new_height, new_width = int(height * scale), int(width * scale)
        frame_resized = cv2.resize(rtmp_frame, (new_width, new_height))

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Run inference on the frame
        results = model(frame_rgb, conf=0.25, iou=0.45)  # Adjust confidence and IOU thresholds

        # Save the frame with detected objects
        detected_filename = f"detected_objects_{frame_count}.jpg"
        results[0].save(filename=detected_filename)
        print(f"Saved detected objects for frame {frame_count} as {detected_filename}")

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"Detected {len(boxes)} objects in frame {frame_count}")
                
                for box in boxes:
                    class_index = int(box.cls)
                    class_name = result.names[class_index]
                    confidence = float(box.conf)
                    
                    print(f"Detected {class_name} with confidence {confidence:.2f}")
                    
                    # Generate JSON for each detected object
                    tStamp = datetime.now().isoformat()
                    devRef = "Captured Image"
                    
                    jLin1 = {
                        "timestamp": tStamp,
                        "device-reference-id": devRef,
                        "details": {
                            "detected_text": "Lorem Ipsum",
                            "detected_object": class_name
                        },
                        "metadata": {
                            "primary_color": "neutral1",
                            "secondary_color": "neutral2",
                            "confidence_primary": confidence,
                            "confidence_secondary": confidence
                        },
                        "context_image": detected_filename,
                        "cropped_image": "image cropped"
                    }

                    jFil = json.dumps(jLin1, indent=4)
                    json_filename = f"_json/jOutFil_{frame_count}_{class_name}_{confidence:.2f}.json"
                    with open(json_filename, "w") as jOutFile:
                        jOutFile.write(jFil)
                    print(f"Saved JSON for {class_name} in {json_filename}")
            else:
                print(f"No objects detected in frame {frame_count}.")

    frame_count += 1

rtsp_cap.release()