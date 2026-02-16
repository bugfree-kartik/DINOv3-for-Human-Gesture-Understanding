# DINOv3 for Human Gesture Understanding

This project compares vision backbones (I-JEPA, Sapiens, DINOv2, DINOv3) and fine-tunes **DINOv3** with LoRA for American Sign Language (ASL) gesture classification on **Sign Language MNIST**.

## Goal

- Train linear probes (logistic regression) on frozen features from multiple vision models.
- Fine-tune DINOv3 with LoRA adapters for 24-class ASL gesture recognition.
- Compare performance and visualize attention / Grad-CAM.

## Dataset

**You do not create the CSVs** — they come from the Sign Language MNIST dataset. Download them once and place in the project root.

1. **Get the files**
   - **Kaggle**: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) — download the dataset and copy into the project root:
     - `sign_mnist_train.csv`
     - `sign_mnist_test.csv`
   - Or use the Kaggle CLI: `kaggle datasets download -d datamunge/sign-language-mnist` then unzip and move the two CSVs to the project root.

2. **Format** (for reference): each row has one column `label` (0–24, letter classes) and 784 pixel columns (28×28 grayscale).

## Setup

1. **Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Working directory**

   - Run the notebook from the project root so paths like `sign_mnist_train.csv` resolve.
   - In Colab, set working directory in the first cell if needed (e.g. upload CSVs and set `%cd` to that folder).

3. **Hugging Face** (for DINOv3 and other models)

   - Log in when prompted: `huggingface_hub.login()` (or set `HF_TOKEN`).

## How to run

1. Open `Project_Notebook.ipynb`.
2. Run cells in order:
   - Data load and preprocessing.
   - Load backbones (Sapiens, I-JEPA, DINOv2, DINOv3) and extract features.
   - Train and evaluate linear classifiers for all four; compare in the table and plots.
   - LoRA section: label remapping, DINOv3 + LoRA, `SignDataset`, training loop, then **save checkpoint** (writes `dinov3_lora_gesture.pth`).
   - Retrain (optional), t-SNE, then Evaluation: attention comparison and Grad-CAM using the saved checkpoint.

## Project layout

- `Project_Notebook.ipynb` – full pipeline (data, backbones, linear probe, LoRA training, evaluation, visualizations).
- `requirements.txt` – Python dependencies.
- `README.md` – this file.

## Notes

- **Comparison**: The “Compare Model Performance” section includes I-JEPA, Sapiens, DINOv2, and DINOv3 (accuracy, precision, recall, F1).
- **LoRA labels**: Remapped labels (`train_labels_remapped`, `test_labels_remapped`) are used only for LoRA training; original `train_labels` / `test_labels` stay unchanged for the linear-probe comparison.
- **Checkpoint**: After LoRA training, a cell saves the model to `dinov3_lora_gesture.pth`. The Evaluation section uses this path (or sets `pth_file = "dinov3_lora_gesture.pth"` if the save cell was skipped).
- **Grad-CAM**: The notebook uses a custom Grad-CAM implementation in the Evaluation section; the `grad-cam` package is not required.
