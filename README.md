# Object Detection System Using YOLOv8

## Project Description

This project is a command-line application written in Python that uses the YOLOv8 model for object detection in images, video files, and live camera streams.

The application allows users to:

- load a pre-trained YOLO model,
- detect objects in a single image,
- detect objects in a video file,
- detect objects in real time using a camera,
- filter detections by confidence threshold,
- draw bounding boxes and labels around detected objects,
- display diagnostic information,
- check the structure of a training dataset,
- fine-tune the model on a custom dataset,
- compare a base model with a fine-tuned model.

The project can be used as a practical example of object detection and as a starting point for developing more advanced computer vision systems.

---

## Requirements

To run the project, Python and the following libraries are required:

```bash
pip install ultralytics opencv-python numpy
