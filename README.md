# ASLens - Sign Language to Text Conversion
ASLens is a deep learning project that convertes sign language gestures into text, using deep recurrent neural networks. 

# Overview
## Dataset

### Step 1: Collecting Data
ASLens uses the **[How2Sign ](https://how2sign.github.io/)** dataset, which consists of video data of sign langauge gestures (ASL language), with the corresponding text translation.
### Step 2: Extracting Hand - Face Keypoints using MediaPipe 
To extract meaningful features for model training, we use **MediaPipe** to extract hand - face keypoints from the video frames. **[MediaPipe ](https://ai.google.dev/edge/mediapipe/solutions/guide)** provides pre-built solutions for detecting hand - face keypoints, which interpret sign language gestures. Due to limited computational resources, a video frame rate of 15fps was used in this project.