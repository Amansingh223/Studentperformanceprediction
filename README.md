# 🎓 Student Performance & Risk Detection System

An advanced Machine Learning web application that predicts a student's final academic performance and automatically detects whether they are at risk of failing — with detailed reasons.


---

## Features

| Feature | Description |
|---------|-------------|
| **Performance Prediction** | Ensemble of models (Random Forest, XGBoost, CatBoost) predicts the final percentage and assigns a grade (A–F) |
| **Pass/Fail Analysis** | Instantly determines if a student is on track to pass or fail |
| **Risk Assessment** | Flags at-risk students with specific reasons like "Low attendance", "Poor quiz scores" |
| **Premium Dark UI** | Fully responsive dark-themed interface with glassmorphism effects and smooth animations |



## Tech Stack

- **Backend:** Python, Flask, Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, CatBoost, GridSearchCV
- **Frontend:** HTML, Vanilla CSS (Dark mode with glassmorphism)
- **Deployment:** Docker, Gunicorn, Render

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Amansingh223/Studentperformanceprediction.git
cd Studentperformanceprediction
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open your browser and navigate to **http://127.0.0.1:5000**

---

## Model Training

The pre-trained model is already included in `artifacts/`. If you want to retrain on the synthetic dataset:

```bash
python -m src.components.data_ingestion
```

This runs the full pipeline — data ingestion → transformation → model training — and saves the best model to `artifacts/model.pkl`.

### Models Evaluated
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- Gradient Boosting Regressor
- Ridge Regression

The best model (by R² score) is automatically selected and saved.

---

## 🌐 Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) and sign up
3. Click **New → Web Service**
4. Connect your GitHub repo
5. Render will detect the `Dockerfile` automatically
6. Set the **Free** plan and click **Deploy**

The app will be live at `https://studentperformanceprediction-1-4n37.onrender.com/`

> **Note:** On Render's free tier, the service spins down after 15 minutes of inactivity. The first request after that takes ~30 seconds to cold-start.

---

## Project Structure

```
Studentperformanceprediction/
├── app.py                          # Flask application
├── Dockerfile                      # Docker container config
├── Procfile                        # Heroku/Render process file
├── render.yaml                     # Render Blueprint config
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── artifacts/
│   ├── model.pkl                   # Trained ML model
│   └── preprocessor.pkl            # Feature preprocessor
├── notebook/
│   ├── generate_dataset.py         # Synthetic dataset generator
│   └── data/
│       └── student_performance_data.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Data loading & splitting
│   │   ├── data_transformation.py  # Feature engineering
│   │   └── model_trainer.py        # Model training & evaluation
│   ├── pipeline/
│   │   └── predict_pipeline.py     # Prediction + risk assessment
│   ├── exception.py                # Custom exception handler
│   ├── logger.py                   # Logging config
│   └── utils.py                    # Helper utilities
└── templates/
    ├── index.html                  # Landing page
    └── home.html                   # Prediction dashboard
```

---
