# Vietnam License Plate Recognition

## Overview

This project develops a system for recognizing Vietnamese license plates using computer vision and deep learning techniques. It combines YOLOv8 for license plate detection and a custom Keras model for character recognition, achieving over 85% accuracy. The system addresses challenges such as varying shooting angles and lighting conditions, making it suitable for real-world applications like traffic monitoring or parking systems.

Key objectives:
- Detect license plate locations in images or videos.
- Recognize characters on Vietnamese license plates, trained on specific fonts and formats.
- Handle environmental variations for robust performance.

Dataset: Custom dataset of Vietnamese license plates (details in `data/` folder or notebook).

## Installation

### Requirements
- Python 3.8+
- Libraries: 
  ```
  ultralytics (for YOLOv8)
  tensorflow
  keras
  opencv-python
  numpy
  matplotlib
  ```

Install dependencies:
```
pip install -r requirements.txt
```


## Key Findings and Results

- **Detection**: YOLOv8n accurately locates plates under various angles and lighting, with mAP > 0.90 on validation set.
- **Recognition**: Custom Keras model trained on Vietnamese fonts recognizes characters with >85% overall accuracy, handling distortions.
- Evaluation: Tested on diverse images (day/night, tilted plates); overcomes common OCR challenges like glare or shadows.
- Example: Input image → Detected plate → Recognized text (e.g., "51F-12345").

Metrics (example from evaluation):
| Metric          | Value    |
|-----------------|----------|
| Detection mAP   | 0.92    |
| OCR Accuracy    | 87%     |
| End-to-End Acc. | 85%+    |

Visualizations: See `results/` for sample outputs with bounding boxes and text overlays.

## Contributors
- Lý Mạnh Thắng (lymanhthang)

## Limitations and Future Work
- Limitations: Performance drops in extreme weather (heavy rain/fog); limited to standard Vietnamese plate formats.
- Future Directions: Integrate real-time video processing (e.g., with OpenCV streams), add support for multi-plate detection, deploy as a web API, or fine-tune with larger datasets.

## References
- YOLOv8: Ultralytics documentation.
- Keras/TensorFlow: Official docs for OCR models.
- OpenCV: For image preprocessing and augmentation.
