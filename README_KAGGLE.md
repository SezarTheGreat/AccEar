# AccEar Kaggle Deployment Guide

This package contains all code files, notebooks, and configuration dependencies needed to run the AccEar cGAN speech reconstruction pipeline on Kaggle.

---

## 📦 File Package Contents

1. `accear_cgan_notebook.ipynb`: Complete Jupyter Notebook ready to upload directly to Kaggle.
2. `accear_cgan_kaggle.py`: Self-contained PyTorch Python script with CLI options for execution.
3. `requirements.txt`: Python package dependencies (`torch`, `torchaudio`, `scipy`, `numpy`, `librosa`, `tqdm`).
4. `StealthyIMU_dataset/*.py`: Preprocessing & metadata helper scripts (`utils.py`, `task4_script.py`, `merge_clean_metadata.py`, `count_medata.py`).

---

## 🚀 How to Upload & Run on Kaggle

### Option A: Direct Notebook Upload (Recommended)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Click **File -> Import Notebook** in the top navigation bar.
3. Drag & drop or select `accear_cgan_notebook.ipynb`.
4. Enable GPU Hardware Accelerator:
   - On the right sidebar, click **Session Options -> Accelerator**.
   - Select **GPU P100** or **GPU T4 x2**.
5. Click **"Run All"**.

### Option B: Kaggle Script / Command Line Upload
1. In your Kaggle notebook cell or terminal, run:
   ```bash
   python accear_cgan_kaggle.py --epochs 200 --batch-size 16 --checkpoint-dir /kaggle/working/checkpoints
   ```
2. Outputs (checkpoints `.pt` and audio samples `.wav`) will automatically be saved to `/kaggle/working/checkpoints/`.

---

## 📊 Dataset Mounting on Kaggle
- If you have uploaded the **StealthyIMU** dataset to Kaggle:
  - Add the dataset to your notebook input via **+ Add Data**.
  - Pass the input path: `--dataset-dir /kaggle/input/stealthyimu/` or update cell 7 in the notebook.
- If no dataset is attached, the code will automatically run using its built-in synthetic Speech & IMU generator so you can test end-to-end training instantly!
