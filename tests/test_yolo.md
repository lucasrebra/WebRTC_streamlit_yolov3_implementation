# Unit Tests Documentation

This document provides details about the unit tests for the YOLO model.

## Tests Overview

### test_yolo_model_loading

- **Description**: This test verifies that the YOLO model is loaded correctly from the configuration and weights files.
- **Assertions**: 
  - Asserts that the `net` object is not `None`.

### test_yolo_detection

- **Description**: This test checks that the YOLO model can perform object detection on a dummy image.
- **Assertions**: 
  - Asserts that the forward pass of the YOLO model returns outputs.
  - Asserts that the length of the outputs is greater than 0.

### test_yolo_classes

- **Description**: This test ensures that the class names are loaded correctly from the `coco.names` file.
- **Assertions**: 
  - Asserts that the length of the `classes` list is greater than 0.

## Running the Tests

To run the unit tests, use the following command:

```sh
pytest tests/test_yolo.py
