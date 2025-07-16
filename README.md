
# Realtime/File Face Blurring

Python Flask & SocketIO Based Application. Blurs your webcam realtime, and able to handle file uploads to detect faces.
## Logic

Uses SocketIO to listen and send live webcam frame data to the python file.\
Uses Flask to listen to the file uploads, retrieving uploaded files, and uploading the processed files for the user to download.
## Installation

Clone The Project

```bash
  git clone https://github.com/nabilarbee/facial-detection-python.git
```

This project was made possible by the following open-source technologies.
Make sure you install the correct versions of each component.
### Core Components
| Component | Version |
| :--- | :--- |
| [Python](https://www.python.org/) | `3.10.9` |
| [Flask](https://pypi.org/project/Flask/) | `3.1.1` |
| [Flask-SocketIO](https://pypi.org/project/Flask-SocketIO/) |`5.5.1` |
| [OpenCV](https://pypi.org/project/opencv-python/) | `4.6.0.66` |
| [MediaPipe](https://pypi.org/project/mediapipe/) | `0.10.21` |
| [Bootstrap](https://getbootstrap.com/) | `5.1.3` |


### References & Learning

*   The computer vision implementation was heavily inspired by the [OpenCV Face Detection Tutorial](https://www.youtube.com/watch?v=DRMBqhrfxXg) created by [ComputerVisionEngineer](https://www.youtube.com/@ComputerVisionEngineer) on YouTube. A big thank you to him for the clear explanation.

## Licensing

This project uses several open-source packages. For a full list of dependencies and their licenses, please see the [NOTICE.md](NOTICE.md) file.