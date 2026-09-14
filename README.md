# Waste Image Classification using PyTorch

A beginner CNN that classifies cardboard, glass, metal, paper, plastic and trash.

## Run in virtual environment

```powershell
python prepare_data.py
python check_project.py
python train.py --epochs 20
python evaluate.py
python predict.py "inputs\mag.jpg"
```

## Files that are saved

- `best_model.pth` in the project root: learned weights, class names, exact split and training history in one file. This is required to predict later and redraw the charts.
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`

The output folder contains only those two charts. Dataset images remain under `data/dataset-resized`, and the magazine example is under `inputs/mag.jpg`.

## Setup on another Windows computer

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

## Code files

| File | Purpose |
|---|---|
| config.py | Paths and training settings |
| prepare_data.py | Download, check and split the data in memory |
| dataset.py | Convert images into tensors and batches |
| model.py | Small CNN architecture |
| train.py | Train and save the best model with split and history |
| evaluate.py | Print test predictions and metrics; save two charts |
| predict.py | Print the prediction for one image |
| check_project.py | Check splits, shapes and learning behavior |


Dataset credit: [TrashNet by Gary Thung and Mindy Yang](https://github.com/garythung/trashnet). 
Framework reference: [PyTorch documentation](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).
