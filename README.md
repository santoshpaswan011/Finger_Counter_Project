✋ Finger Counter using MediaPipe & OpenCV

A real-time finger counting application that uses your webcam to detect a hand and count how many fingers are raised. It is built with MediaPipe Tasks API and OpenCV.

📌 Overview

The project detects 21 hand landmarks in each video frame and compares the positions of the fingertips with their joints to decide whether each finger is up or down. The total count is then shown live on the screen.

✨ Features
Real-time hand detection through the webcam
Counts fingers from 0 to 5 (supports one or both hands)
Draws hand landmarks and connections on the video feed
Shows the finger count on screen as an overlay
Lightweight and runs smoothly on a normal CPU
🛠️ Tech Stack
Python
MediaPipe (Tasks API, Hand Landmarker)
OpenCV
NumPy
⚙️ How It Works
Capture video frames from the webcam using OpenCV.
Pass each frame to the MediaPipe Hand Landmarker to get 21 landmarks.
Compare the tip landmark of each finger with the joint below it. A tip above its joint means the finger is up.
Check the thumb separately using the x-axis, since it moves sideways.
Add up the raised fingers and display the count on the frame.
🚀 Installation & Usage
bash
# Clone the repository
git clone https://github.com/<your-username>/finger-counter.git
cd finger-counter

# Install dependencies
pip install opencv-python mediapipe numpy

# Run the project
python main.py

Press q to quit the webcam window.

Windows tip: If the camera does not open, use cv2.VideoCapture(0, cv2.CAP_DSHOW).

📂 Project Structure
finger-counter/
├── finger_counter.py
├── hand_landmarker.task
├── requirements.txt
└── README.md
🔮 Future Improvements
Gesture recognition (thumbs up, peace sign, etc.)
Volume or brightness control using finger count
Support for multiple hands with separate counts
👨‍💻 Author

Santosh Paswan
B.Tech CSE (IoT) | Interested in AI, Machine Learning & Computer Vision
