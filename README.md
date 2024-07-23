# Person Detection in Restricted Zones Using YOLOv10

This project uses a pre-trained YOLOv10 model and OpenCV to process video frames, detect persons, and identify their presence within a defined restricted area. The output is a video with annotations highlighting the detected persons and indicating if they are within the restricted zone.

## Features

- **Object Detection**: Utilizes YOLOv10 to detect persons in each video frame.
- **Restricted Area Monitoring**: Checks if detected persons are within a specified rectangular area.
- **Real-time Annotations**: Draws bounding boxes around detected persons and a rectangle representing the restricted area.
  - Red bounding box: Person is inside the restricted area.
  - Green bounding box: Person is outside the restricted area.
- **Text Overlay**: Displays the count of persons in the restricted area with a background for better visibility.
- **Video Processing**: Processes input video frame by frame and saves the annotated output video.

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HussainNasirKhan/Person-Detection-in-Restricted-Zones-Using-YOLOv10.git
   cd Person-Detection-in-Restricted-Zones-Using-YOLOv10

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage

1. **Set the restricted area coordinates**:
   Modify the `restricted_area` variable in the script with your desired coordinates:
   ```python
   restricted_area = (x1, y1, x2, y2)

2. **Run the script**:
   ```bash
   python app.py

3. The processed video will be saved as **output.mp4** in the current directory.

## Contributing

Feel free to open issues or submit pull requests for any improvements or new features. Your contributions are welcome!
