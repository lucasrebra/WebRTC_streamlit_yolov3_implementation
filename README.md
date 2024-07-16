# WebRTC Object Detection with YOLO

This project demonstrates real-time object detection using YOLO (You Only Look Once) integrated with WebRTC, allowing video streaming and processing directly in the browser via a Streamlit application.

## Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Files](#files)
- [Usage](#usage)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7 or higher
- Streamlit
- OpenCV
- NumPy
- `streamlit-webrtc` library
- `av` library

### Steps

1. Clone the repository:

    ```sh
    git clone <repository_url>
    cd my_webrtc_yolo_project
    ```

2. Create a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:

    ```sh
    pip install opencv-python-headless streamlit-webrtc av numpy streamlit
    ```

4. Download YOLOv3 weights, configuration file, and COCO class names:

    - [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    - [yolov3.weights]([https://pjreddie.com/media/files/yolov3.weights](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights))
    - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

    Place these files in your project directory.

## Running the Application

1. Start the Streamlit application:

    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to see the application in action.


## Files

- `app.py`: Main application file running the Streamlit server and handling the WebRTC video stream and YOLO object detection.
- `yolov3.cfg`: YOLOv3 configuration file.
- `yolov3.weights`: YOLOv3 pre-trained weights.
- `coco.names`: List of class names used by YOLO.
- `requirements.txt`: List of dependencies.
- `README.md`: Project description and instructions.
- `setup.py`: Packaging configuration.
- `tests/`: Directory containing all test files.

### Unit Tests

Unit tests are located in the `tests/test_yolo.py` file. These tests include:

- **test_yolo_model_loading**: Verifies that the YOLO model loads correctly.
- **test_yolo_detection**: Checks that the YOLO model performs object detection on a dummy image.
- **test_yolo_classes**: Ensures that the class names are loaded from the `coco.names` file.

For more details, see [Unit Tests Documentation](docs/tests/test_yolo.md).

## Usage

1. After starting the application, grant the necessary permissions for the browser to access your camera.
2. The video stream from your webcam will be processed in real-time to detect objects using YOLO.
3. Detected objects will be highlighted with bounding boxes and labels in the video feed.

## Notes

- The application runs on HTTP by default. For better security and to avoid permission issues with accessing the camera, consider running the application over HTTPS.
- You can use a self-signed certificate for local testing or get a valid certificate from a certificate authority for production use.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


This project is licensed under the MIT License. See the `LICENSE` file for details.
