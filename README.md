# DL-ESC50

This repository contains a small deep learning project for environmental sound classification on the ESC-50 dataset. The work focuses on identifying 5-second audio clips into 50 sound classes such as dog barking, sirens, rain, firework bursts, crying babies, and glass breaking.

## Project overview

The project explores three modeling approaches in the notebook:

- A compact CNN trained from scratch
- An improved CNN with stronger training practices
- A pretrained Audio Spectrogram Transformer (AST) fine-tuned on ESC-50

The notebooks compare model performance on a held-out test set and summarize the results with evaluation metrics such as accuracy and F1-score.

## Repository structure

```text
DL-ESC50/
├── README.md
├── requirements.txt
├── .venv/
├── ESC-50-master/
│   ├── README.md
│   ├── audio/
│   ├── meta/
│   └── ...
├── notebook/
│   ├── ESC50_Simple_Project.ipynb
│   └── esc50_dl_model_testings.ipynb
└── outputs/
```

## Environment setup

This project was tested with Python 3.13 and uses TensorFlow, PyTorch, librosa, scikit-learn, pandas, and related audio-processing libraries.

```bash
cd /Volumes/Coding Work Place/DL-ESC50
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset download and setup

If the local dataset folder is not present, download the ESC-50 archive and extract it:

```bash
curl -L --retry 5 --retry-delay 2 -o ESC-50.zip \
  https://github.com/karoldvl/ESC-50/archive/master.zip

unzip ESC-50.zip
```

The project expects the extracted dataset in a folder named `ESC-50-master`, which contains:

- `audio/` – waveform files for all ESC-50 clips
- `meta/esc50.csv` – metadata with labels and fold assignments

Official dataset repository: https://github.com/karoldvl/ESC-50

## Running the notebooks

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open:

- `notebook/ESC50_Simple_Project.ipynb`
- `notebook/esc50_dl_model_testings.ipynb`

These notebooks walk through:

1. Data loading and class inspection
2. Audio preprocessing and feature extraction
3. Model training and validation
4. Evaluation and comparison of results

## Notes

- The dataset is intended for research and educational use.
- Some notebooks expect the project root and dataset folder to be organized as shown above.
- For GPU acceleration, using Google Colab or a machine with CUDA-capable hardware is recommended.

## License

The ESC-50 dataset is distributed under its own licensing terms. Please review the dataset repository and source materials before reusing or publishing results.
