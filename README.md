# Plastic Packaging Detection and Material Classification

This project combines **YOLO** for object detection and **Xception** for material classification of plastic packaging into three manufacturing types:

- **TF** – Thermoformed  
- **IM** – Injection moulded  
- **FE** – Foil extrusion  

The final system detects multiple plastic items in a full image and assigns a material label to each object.

---

## Repository Contents

- `YOLO.ipynb`  
  Training and evaluation of a custom YOLOv11n model on the plastic packaging dataset.  
  Uses images and labels from the lab server (e.g. `/VOL_D/teresa.w/Dataset_YOLO/dataset`),  
  and saves weights to `/VOL_D/asim.n/runs/train/yolo11n_custom/weights/best.pt`.

- `xception_train.ipynb`  
  Fine-tuning an Xception classifier on cropped object images from  
  `/VOL_D/teresa.w/Dataset_Xception/full_dataset`.  
  Class mapping:  
  - 0 → Thermoformed  
  - 1 → Injection moulded  
  - 2 → Foil extrusion  
  Saves the trained model to  
  `/VOL_D/asim.n/xception_xray_project/xception_full_dataset.h5`.

- `YOLO+Xception.ipynb`  
  Integrated inference pipeline:
  1. Runs YOLO on full validation images from `/VOL_D/teresa.w/Dataset_YOLO/dataset/images/val`.
  2. Crops each YOLO bounding box.
  3. Classifies each crop with the Xception model (TF / IM / FE).
  4. Applies a confidence threshold and optionally outputs **"Unknown"** for low confidence.
  5. Draws bounding boxes and labels on the original image and saves results to  
     `/VOL_D/asim.n/yolo_xception_outputs`.
  6. Evaluates the combined system and generates confusion matrices and recall–confidence curves.

- `confusion_matrix.png`, `confusion_matrix_normalized.png`  
  Confusion matrices for the combined YOLO+Xception system.

- `BoxR_curve.png`  
  Recall–confidence curve showing the trade-off between confidence threshold and recall per class.

- `results.csv`  
  CSV file with per-image / per-class evaluation metrics (used to generate the plots).

---

## Data and Weights (Not Included)

The datasets and trained weights are stored on the lab server and are **not** part of this repository:

- YOLO dataset: `/VOL_D/teresa.w/Dataset_YOLO`
- Xception dataset: `/VOL_D/teresa.w/Dataset_Xception`
- YOLO weights: `/VOL_D/asim.n/runs/train/yolo11n_custom/weights/best.pt`
- Xception model: `/VOL_D/asim.n/xception_xray_project/xception_full_dataset.h5`

To reproduce the experiments, run the notebooks on the lab environment where these paths are available.
