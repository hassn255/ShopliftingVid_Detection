# 🎥 Person Cropping from Video (YOLOv8)

## Overview
This script extracts and crops **person regions** from videos using a **YOLOv8 model**, producing uniform `256×256` clips centered on the detected person.  
Useful for preprocessing datasets in **action recognition** or **behavior detection**.

---

## How It Works
1. **Detection:** YOLOv8 detects people in each frame. The largest bounding box is chosen if multiple are found.  
2. **Valid Range:** Frames before the first or after the last detection are replaced with **black frames**.  
3. **Interpolation:** Missing boxes between valid detections are linearly **interpolated** to keep motion smooth.  
4. **Cropping:** Each person box is slightly **expanded** and resized to `(256, 256)` before saving.

---

## Requirements
```
pip install ultralytics opencv-python numpy
```

## Usage

python
Copy code

```
from ultralytics import YOLO
from crop_person import crop_person_from_video  # your script

model = YOLO("yolov8n.pt")

crop_person_from_video(
    model=model,
    input_path="Shop DataSet/train/theft/video1.mp4",
    output_path="Processed/train/theft/video1_cropped.mp4",
    expand_ratio=0.3,
    output_size=(256, 256)
)
```
## Parameters

|Name | Description | Default |
|-----|-------------|---------|
|`model` | YOLOv8 model instance | — |
|`input_path` |	Input video path | — |
|`output_path` | Output video path | — |
|`expand_ratio` | Box expansion factor | 0.3 |
|`output_size` | Cropped video size	| (256, 256)|
```
## Dataset Example

Input:

```css
Copy code
Shop DataSet/
 ├── train/
 │   ├── normal/
 │   ├── theft/
```
Output:

```css
Copy code
Processed/
 ├── train/
 │   ├── normal/
 │   ├── theft/
 ```

## Notes
- Black frames represent segments where no person was present.

- Interpolation fills only middle missing frames, not start/end.

- Works best for clips with one main person.

- FPS is preserved from the original video.