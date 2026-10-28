# California Housing Price Prediction App

**Junior Machine Learning Portfolio Project 06/06**  
**Author:** Moe Htet Ar Kar (Phoe Cho)

A regression dashboard that uses California census block-group features to estimate median house value with Linear Regression.

> **Scope:** A learning and portfolio application built with a small, understandable Python script. It is not a production or safety-critical system.

## Dashboard preview

![Illustrative dashboard design preview](screenshots/dashboard-preview.png)

*This image is a design mockup for layout reference, not a screenshot captured from a running app. Replace it with your own live screenshot when you run the project; save that capture as `screenshots/output.png`.*

## What the project demonstrates

- Building an interactive user interface with Streamlit.
- Preparing input features in a consistent order and passing them to a scikit-learn model.
- Training and caching a model to avoid retraining on every Streamlit rerun.
- Presenting model output and interpreting it with appropriate caveats.
- Organizing a small Python project with isolated dependencies and Git hygiene.

## Model and data

**Model / workflow:** LinearRegression; random 80/20 train/test split; MSE and R² reported on the held-out split.

**Training data:** The scikit-learn California Housing dataset: 20,640 block-group observations and eight numeric features. The target MedHouseVal is measured in $100,000 units.

## Output design

A slate-and-teal real-estate dashboard with sliders for the eight census features, estimated median house value formatted in USD, and regression metrics from a held-out test split.

The live application is designed to show the entered features, the model output, and an explanation of what the result means. A prediction is only as reliable as its data, labels, features, and evaluation method.

## Technology stack

- Python 3.11
- Streamlit
- pandas / NumPy where used by this project
- scikit-learn
- TensorFlow/Keras where required by this project

## Run locally

### 1. Prerequisite

Install **64-bit Python 3.11**. Confirm it is available in your terminal:

```bash
python --version
```

On Windows, if the `python` command points to another interpreter, use `py -3.11` in place of `python` when creating the environment.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

**Data note:** Scikit-learn downloads the California Housing dataset on first use and caches it locally. An internet connection may be required on the first launch.

### 4. Start the application

Run this command from the same folder as `requirements.txt`:

```bash
streamlit run california_housing_app.py
```

Streamlit prints a local URL in the terminal (normally `http://localhost:8501`). Open that address in your browser. To stop the app, press `Ctrl+C` in the terminal.

## Project folder structure

```text
06_california_housing_app/
├──.venv/
├── california_housing_app.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    ├── output.png          
```

## Current limitations

This public dataset comes from the 1990 U.S. Census and is not a current property valuation dataset. The model uses a simple linear relationship and may produce inaccurate or out-of-range estimates.
The target is capped in the source dataset. The dataset may need to download from the internet on first run. Do not use as a real-estate appraisal or financial decision.

## Future improvements

1. Replace generated or toy data with a documented, appropriate dataset and record its source and license.
2. Add a repeatable evaluation process, appropriate metrics, baseline comparisons, and model-validation tests.
3. Save a trained model only after validating it, then separate training from app inference if the project grows.
4. Add input validation, clear error messages, and a simple automated test suite.


## Portfolio summary

This project demonstrates a junior-level end-to-end workflow: a Python ML model, a simple interactive front end, input handling, result presentation, dependency isolation, and honest documentation of limitations.
It should be presented as a learning prototype rather than a production-grade system.

## References

- [scikit-learn California Housing dataset reference](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [California Housing dataset description](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/descr/california_housing.rst)
- [scikit-learn Linear Regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
