# Facial Emotion Recognition Using Machine Learning, Deep Learning and Computer Vision

**Course project:** BS AI, 5th Semester, Integrated AI Project (Machine Learning, Deep Learning, Computer Vision)
**Milestone:** 2 (Dataset, Preprocessing and Initial Model)
**Supervisor:** Sir Aqib Mehdi

## Team

| Member | Area | Main file |
| --- | --- | --- |
| Mugheeza | Machine Learning | `Mugheeza_ML_HOG_Baseline.ipynb` |
| Kainat Ashraf | Deep Learning | `Kainat_DL_CNN_Baseline.ipynb` |
| Wardah Amir | Computer Vision | `Wardah_CV_Pipeline.ipynb` |

Each member also has an individual repository with the same work and its own detailed README.

## Problem

Automatically classify a face image into one of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral. The project compares three approaches on the same dataset and joins them in one working pipeline.

```
Photo -> OpenCV face detection -> 48x48 grayscale crop -> CNN -> emotion + probabilities
```

## Dataset

FER-2013 (Kaggle): https://www.kaggle.com/datasets/msambare/fer2013
35,887 grayscale images (48 x 48), 7 classes. Split used by all members: 24,402 train, 4,307 validation and 7,178 test images. The dataset is not stored in this repository because of its size; the notebooks download it automatically.

## Files in this repository

| File | Content |
| --- | --- |
| `Mugheeza_ML_HOG_Baseline.ipynb`, `ml_results.csv`, `ml_*.png` | Machine Learning part: HOG features, SVM, KNN, Random Forest, results table, class chart, confusion matrices |
| `Kainat_DL_CNN_Baseline.ipynb`, `dl_*.png` | Deep Learning part: CNN, training curves, confusion matrix |
| `Wardah_CV_Pipeline.ipynb`, `Emotion_App_Gradio.ipynb`, `cv_*.png` | Computer Vision part: face detection pipeline, Gradio app, test-photo results |
| `GRUP-M Proposal.pdf`, `GRUP-M Report.docx` | Project proposal and Milestone 2 report |
| `requirements.txt` | Python packages |

## Results (FER-2013 test set, 7,178 images)

| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| HOG + SVM (8,000 training images) | 47.84% | 53.79% | 42.40% | 44.53% |
| HOG + KNN | 46.06% | 44.49% | 42.16% | 42.20% |
| HOG + Random Forest | 45.08% | 56.03% | 38.67% | 41.04% |
| **CNN** | **60.66%** | **62.18%** | **51.92%** | **52.48%** |

Precision, recall and F1 are macro averages over the 7 classes. The CNN is almost 13 points more accurate than the best classical model.

**Working prototype:** on 5 clear test photos, the full pipeline (face detection + CNN) gave the expected emotion for 4 of 5 photos, and the pretrained DeepFace model for 3 of 5. Expected emotions are the group's own reading of the faces, so this is only a small demonstration.

![Pipeline on five photos](cv_pipeline_five_photos.png)

## How to run

All notebooks run in Google Colab (File > Upload notebook, then run the cells from top to bottom):

1. `Mugheeza_ML_HOG_Baseline.ipynb`
2. `Kainat_DL_CNN_Baseline.ipynb` (use a GPU runtime)
3. `Wardah_CV_Pipeline.ipynb` (upload `emotion_cnn.keras` and `classes.json`, the files saved by the CNN notebook)

**Note:** OpenCV 5 does not include the Haar Cascade face detector, so use OpenCV 4: `pip install "opencv-python-headless<5"`.

## Problems encountered

- Class imbalance: Disgust has only 436 training images, so it is poorly recognized (CNN recall 0.11).
- Fear, Sad, Angry and Surprise are often confused with each other.
- The SVM was trained on 8,000 images only, to save time.
- The first face detector settings produced extra false faces; now only the largest face is kept.

## Planned improvements (Milestone 3)

- Class weights, hyperparameter tuning, transfer learning and SVM on the full training set.
- A complete web application (Flask backend and HTML/JavaScript frontend) with photo upload and live camera.
- Video and webcam demo, and a larger face-detection test.
