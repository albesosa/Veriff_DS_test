# 🎥 Multiple people face spoof detection in videos

This project provides a complete pipeline for **video-based face spoof detection**, including:

- **Face detection** (YOLOv3 or MediaPipe)
- **Spoof detection** (DeepPixBiS or CDCN++)
- **Face tracking** (SORT)
- **Per-frame classification** of real vs fake faces
- **Final video-level label prediction**: Whether a video shows more than one real face.

---

## 📁 Project Structure
```
├── data/                               # Folder containing video files to be processed
├── output_data/                        # Folder containing csv files with results
├── models/                             # Pretrained models (YOLOv3-tiny, DeepPixBiS, etc.)
│   ├── yolo_face/
│   └── deeppixbis/
├── discovery/                          # Scripts used for discovery and testing 
├── src/                                # Source code
│   ├── face_detector.py                # Face detection (YOLOv3 / MediaPipe)
│   ├── spoof_detector.py               # Spoof detection logic
│   ├── face_tracker.py                 # Face tracking (SORT)
│   ├── video_processor.py              # Main processing pipeline
│   ├── main_video_processing.py        # CLI interface
│   ├── predict_all_videos_labels.py    # Script to predict labels for all processed videos    
│   ├── calculate_metrics.py            # Metrics calculation (precision, recall, F1-score)
│   └── utils.py                        # Logging setup and helper functions
├── pyproject.toml                      # Poetry dependency and project config
└── README.md
```

## 🛠 Installation
This project uses Poetry to manage dependencies.
To get started:

1. Install Poetry (if not already installed):
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Clone the repository
    ```
    git clone
    ```
3. Install dependencies with Poetry
   ```
   poetry install
   ```
4. Activate the virtual environment
   ```
    poetry shell
    ```
   
## ▶️ How to Run

### ✅ Process a Single Video
```
python -m src.main_video_processing \
    --video_to_process data/veriff6.mp4 \
    --show
```
### ✅ Process All Videos in a Directory
```
python -m src.main_video_processing \
    --video_to_process all \
    --data_dir data
```

### Important
To run the above commands, make sure you are in the project root 
directory where `pyproject.toml` is located.

### 🧠 CLI Arguments:
| Argument                  | Description                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| `--video_to_process`      | Path to video file or `"all"` to process all videos in `--data_dir`  |
| `--data_dir`              | Directory where videos are located (used when processing all videos) |
| `--threshold`             | Spoof detection threshold (default: 0.25)                            |
| `--show`                  | Flag to display video frames while processing                        |
| `--face_detector_method`  | Face detection method (`yolov3` or `mediapipe`)                      |
| `--spoof_detector_method` | Spoof detection model (`deeppixbis`, `cdcnpp`)                       |
| `--face_tracker_method`   | Face tracker (`sort`)                                                |
| `--verbose`               | If set, enables verbose logging                                      |


By default models used are:
- **Face Detector**: YOLOv3-tiny
- **Spoof Detector**: DeepPixBiS
- **Face Tracker**: SORT

If you want to use different models, you can specify them using the CLI arguments.
Example:
```
python -m src.main_video_processing \
    --video_to_process data/veriff6.mp4 \
    --face_detector_method mediapipe \
    --spoof_detector_method cdcnpp \
    --face_tracker_method sort
```

## 📊 Metrics Calculation

To calculate metrics like precision, recall, and F1-score for the processed videos:
1- First loop over all videos and generate predictions:
```
python -m src.predict_all_videos_labels --output_data_dir output_data --models_used "yolov3_deeppixbis_sort"
```
2- Then calculate metrics based on the predictions vs labels:
```
python src/calculate_metrics.py
```

## 💾 Output
- CSV files will be saved for each video containing per-frame results: face bounding boxes, spoof labels, etc.
- Final label prediction (0 or 1) indicates if more than one real person was detected in the video.
- CSV with merged results, i.e., video_name, actual_label, predicted_label and models used,  
will be saved in `output_data/merged_results.csv`.
- Metrics will be printed to the console


