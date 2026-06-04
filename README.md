# Student Performance & Risk Detection System

This project is an advanced Machine Learning web application designed to predict a student's final academic performance (percentage and grade) and automatically detect whether they are at high, medium, or low risk of failing.

Unlike simple predictors, this system analyzes multiple actionable data points, including:
- Attendance percentage
- Assignment and quiz scores
- Internal marks
- Weekly study hours
- Previous semester CGPA

## Features
- **Performance Prediction:** Uses an ensemble of models (Random Forest, XGBoost, CatBoost) to predict the final percentage and assign a grade (A-F).
- **Pass/Fail Analysis:** Instantly determines if a student is on track to pass.
- **Risk Assessment System:** Intelligently flags students and provides specific reasons (e.g., "Low attendance", "Poor assignment completion") so teachers or parents can intervene early.
- **Premium UI:** A fully responsive, dark-themed UI built with HTML/CSS and Jinja2, featuring smooth animations and progress indicators.

## Tech Stack
- **Backend:** Python, Flask, Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, CatBoost
- **Frontend:** HTML, Vanilla CSS (Dark mode with glassmorphism effects)

## How to Run Locally

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000`

## Model Training
If you want to retrain the model on the synthetic dataset, simply run:
```bash
python -m src.components.data_ingestion
```
This will trigger the data transformation pipeline and train all the regression models, automatically saving the best-performing one to the `artifacts/` folder.
