# Football Player Re-Identification and Tracking

This project detects, tracks, and re-identifies football players using a custom-trained YOLOv8 model and DeepSORT tracker. The system ensures players retain consistent IDs even after going out of view and reappearing.

## Files Included

- `track_players.py` – Main Python script
- `best.pt` – YOLOv8 trained model for player and ball detection
- `tracked_output.mp4` – Output video with tracking and IDs
- `requirements.txt` – List of dependencies
- `README.md` – Project setup and usage guide
- `report.pdf` – Brief report of approach and challenges

## ⚙️ Setup Instructions

1. Make sure you have **Python 3.8+** installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

# How to Run
1. Place your input video (e.g., 15sec_input_720p.mp4) in the same folder.
2. Run the script:  python track_players.py

Output video will be saved as tracked_output.mp4.

3. Dependencies
ultralytics
opencv-python
torch
deep_sort_realtime
numpy

# Install them using: pip install ultralytics opencv-python numpy matplotlib torch

# Model
The best.pt file is a fine-tuned YOLOv8 model trained to detect:
 Players
 Ball
 Referees
Ensure this file is placed in the root directory before running.


# Credits
 YOLOv8 by Ultralytics
 DeepSORT for tracking logic
 Asignment provided by Liat.ai
