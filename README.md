# AiMlProject

This repository contains example machine learning and reinforcement learning experiments using the Pima Indians Diabetes dataset and a custom Gym environment for canteen queue management.

## Project overview

- diabetes.csv / diabetes-checkpoint.csv
  - The Pima Indians Diabetes dataset (CSV format). It contains the following columns:
    - Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
  - `Outcome` is the target: 1 indicates diabetes, 0 indicates no diabetes.

- ml_model.ipynb (and its checkpoint)
  - A Jupyter notebook that trains an SVM classifier (linear kernel) on the diabetes dataset.
  - The notebook shows preprocessing with StandardScaler, an 80/20 train/test split, training and evaluation.
  - Example reported results (from the notebook run):
    - Accuracy: 75.97%
    - Classification report (test set): precision/recall/f1 for classes 0 and 1 included in the notebook output.

- Untitled.ipynb (and its checkpoint)
  - A notebook that implements a custom Gymnasium environment `CanteenQueueEnv` and trains a PPO agent (stable-baselines3).
  - The environment simulates arrivals and service during a 60-minute lunch rush and penalizes long queues and staff costs.
  - The notebook includes package installation logs and a short simulated run output.

- .ipynb_checkpoints/
  - Notebook checkpoint files and CSV checkpoint copies are present. It's recommended to remove these from the repo and add a `.gitignore` entry to avoid committing checkpoint files.

## How to run

1. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows PowerShell
```

2. Install required packages

For the SVM notebook (ml_model.ipynb):

```bash
pip install pandas scikit-learn numpy jupyter
```

For the RL notebook (Untitled.ipynb):

```bash
pip install gymnasium stable-baselines3 torch numpy pandas jupyter
```

You can combine the packages into a `requirements.txt` if you prefer.

3. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open `ml_model.ipynb` to run the diabetes SVM example, or open `Untitled.ipynb` to run the RL simulation.

Notes when running:
- `ml_model.ipynb` expects `diabetes.csv` to be in the same directory (or update the path in the notebook).
- `Untitled.ipynb` installs packages inside a notebook cell in the provided version; prefer installing them before running the notebook.

## Recommendations / Next steps

- Remove `.ipynb_checkpoints/` from the repository and add the following to `.gitignore`:

```
.ipynb_checkpoints/
```

- Add a `requirements.txt` listing the project dependencies so others can install them with `pip install -r requirements.txt`.

- Consider moving scripts out of notebooks into `.py` files for reproducible runs (for example, a `train_svm.py` and `train_ppo.py`).

- Add data validation and preprocessing steps in the SVM notebook (handle zero values for features like Glucose, BloodPressure, SkinThickness, Insulin, BMI which may indicate missing data in this dataset).

- Add model saving and a small evaluation script to load a trained model and run predictions on new examples.

## License

This repository has no license specified. If you want to allow others to use or contribute, add a `LICENSE` file (for example, MIT License).

## Contact

Repository owner: @Ravibbk

