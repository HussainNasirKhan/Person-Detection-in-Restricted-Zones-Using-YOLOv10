import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10x.pt")

# Open the input video file
input_video_path = "input_video.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define the restricted area (x1, y1, x2, y2)
restricted_area = (700, 500, 1100, 900)  # Modify these coordinates as needed

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)
    
    # Get the first result (assuming one result per frame)
    result = results[0]

    # Filter results to only include persons (class ID 0)
    person_boxes = result.boxes[result.boxes.cls == 0]

    # Count the number of persons in the restricted area
    persons_in_area = 0

    # Draw the restricted area on the frame
    cv2.rectangle(frame, (restricted_area[0], restricted_area[1]), (restricted_area[2], restricted_area[3]), (255, 0, 0), 2)

    # Draw person boxes on the frame and check if they are in the restricted area
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Check if the person is in the restricted area
        if (restricted_area[0] < x1 < restricted_area[2] and
                restricted_area[1] < y1 < restricted_area[3]) or (
                restricted_area[0] < x2 < restricted_area[2] and
                restricted_area[1] < y2 < restricted_area[3]):
            # Draw a red bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            persons_in_area += 1
        else:
            # Draw a green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the number of persons in the restricted area with a background
    text = f'Persons in restricted area: {persons_in_area}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2  # Increase text size
    thickness = 3  # Increase thickness

    # Get the width and height of the text box
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    # Set the text start position
    text_x = 100
    text_y = 100

    # Create a background rectangle for the text
    cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 20), 
                  (text_x + text_size[0] + 20, text_y + 20), (0, 0, 0), -1)

    # Add text on top of the background rectangle
    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)  # Yellow text
    
    # Write the frame to the output video
    out.write(frame)

# Release the video objects
cap.release()
out.release()

print(f"Filtered video saved as {output_video_path}")
