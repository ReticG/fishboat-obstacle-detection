README.md
# Fishboat Obstacle Detection

This project aims to detect obstacles in fishboat surveillance videos using a stereo camera setup. It includes depth estimation and object detection using pre-trained models.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Demo](#running-the-demo)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Notebook (optional, for running the demo notebook)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fishboat-obstacle-detection.git
   cd fishboat-obstacle-detection

2.  Install dependencies:
pip install -r requirements.txt

3.  Download the pre-trained YOLOv8 model (if not already present in the models/ directory):
mkdir -p models
wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

Running the Demo
1.  Run the demo notebook:
jupyter notebook notebooks/demo.ipynb

2.  Follow the instructions in the notebook to test the depth estimation and object detection functionalities.
Project Structure
fishboat-obstacle-detection/
├── data/
│   ├── left_images/
│   ├── right_images/
│   └── depth_maps/
├── models/
│   └── yolov8n.pt
├── notebooks/
│   └── demo.ipynb
├── src/
│   ├── depth_estimation.py
│   ├── object_detection.py
│   └── utils.py
├── README.md
├── requirements.txt
└── LICENSE

•  data/: Contains the left and right images for stereo vision, and the resulting depth maps.
•  models/: Contains pre-trained model files.
•  notebooks/: Contains Jupyter notebooks for interactive demos.
•  src/: Contains Python scripts for depth estimation, object detection, and utility functions.
•  README.md: This file, providing an overview and instructions for the project.
•  requirements.txt: Lists the Python dependencies required to run the project.
•  LICENSE: The license under which the project is released.
Dependencies
•  OpenCV: For image processing and computer vision tasks.
•  NumPy: For numerical operations.
•  PyTorch: For deep learning tasks.
•  Ultralytics YOLO: For object detection using pre-trained YOLO models.
License
This project is licensed under the MIT License - see the LICENSE LICENSE file for details.
Acknowledgments
•  Ultralytics YOLO https://github.com/ultralytics/ultralytics for the pre-trained YOLO models.
•  OpenCV https://opencv.org/ for the computer vision library.
•  PyTorch https://pytorch.org/ for the deep learning framework.

