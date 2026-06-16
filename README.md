# CreditSight — Alternative Credit Scoring

A machine learning system that scores loan applicants using 
behavioral signals instead of traditional bureau history, targeting 
India's 190 million credit-invisible population.

## What this does
- LightGBM model trained on 89 features including behavioral signals
  (payment regularity, credit diversity, application velocity)
- Fairness audit using Fairlearn across geographic segments  
- FastAPI backend with real-time credit scoring
- Clean frontend demo for live credit assessment

## Tech stack
LightGBM · FastAPI · Fairlearn · SHAP · Python · HTML/CSS/JS

## Run locally
pip install fastapi uvicorn joblib lightgbm pandas pydantic
uvicorn main:app --reload
Then open index.html in your browser.

## Research
Targeting FAccT 2026 submission.
Dataset: Home Credit Default Risk (307,511 applications)
