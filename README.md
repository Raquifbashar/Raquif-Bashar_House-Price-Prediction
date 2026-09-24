# 🏠 House Price Prediction using Machine Learning

**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares**

**Author:** Raquif Bashar

---

## 📌 Project Description

This project builds a machine learning regression model that predicts the
**median house value** of a district in California based on features such as
location, housing age, room counts, population, household income, and
proximity to the ocean.

The workflow covers the complete data analytics lifecycle:

1. Data loading and inspection
2. Exploratory Data Analysis (EDA) with visualizations
3. Data cleaning and feature engineering
4. Preprocessing pipeline (imputation, scaling, one-hot encoding)
5. Training and comparing multiple regression models
6. Hyperparameter tuning of the best model
7. Model evaluation and feature importance analysis
8. Saving the final model for reuse

## 📊 Dataset

- **Name:** California Housing Prices
- **Description:** Derived from the 1990 U.S. Census; each row represents one
  California district (a block group), with 20,640 records and 10 columns.
- **Dataset link:** https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
- A local copy is also included in this repository at `data/housing.csv`.

| Column | Description |
|---|---|
| `longitude`, `latitude` | Geographic coordinates of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms`, `total_bedrooms` | Total rooms / bedrooms in the district |
| `population`, `households` | Population and number of households |
| `median_income` | Median household income (in tens of thousands of USD) |
| `ocean_proximity` | Categorical distance category from the ocean |
| `median_house_value` | **Target** — median house value in USD |

## 🛠️ Technologies Used

- **Language:** Python 3
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib
- **Environment:** Jupyter Notebook
- **Models trained:** Linear Regression, Decision Tree, Random Forest,
  Gradient Boosting (final model: **tuned Random Forest Regressor**)

## 📈 Key Results

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **Random Forest** | 31,926 | 49,834 | **0.810** |
| Gradient Boosting | 36,404 | 53,505 | 0.782 |
| Linear Regression | 49,645 | 69,127 | 0.635 |
| Decision Tree | 43,093 | 69,861 | 0.628 |

After hyperparameter tuning, the final **Random Forest** model achieved an
**R² score of 0.812** on the held-out test set. The most influential features
were **median income**, **ocean proximity (inland)**, and
**population per household**.

## 📂 Project Structure

```
├── Raquif_Bashar_HousePricePrediction.ipynb   # Main project notebook (code + outputs)
├── Raquif_Bashar_ProjectReport.docx           # Full project report
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
├── house_price_model.pkl                      # Saved trained model pipeline
├── data/
│   └── housing.csv                            # Dataset (California Housing Prices)
└── images/                                    # Exported charts from the notebook
```

## ⚙️ Setup & Run Instructions

1. **Clone / download** this project folder.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Raquif_Bashar_HousePricePrediction.ipynb
   ```
5. **Run all cells** (`Kernel → Restart & Run All`) to reproduce the full
   analysis, charts, and model training from scratch.

### Using the saved model for a new prediction

```python
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

new_data = pd.DataFrame([{
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
    "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
    "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY",
    "rooms_per_household": 880.0 / 126.0, "bedrooms_per_room": 129.0 / 880.0,
    "population_per_household": 322.0 / 126.0,
}])

print(model.predict(new_data))
```

## 🚀 Future Scope

- Add real-world features such as crime rate and school ratings.
- Try advanced models like XGBoost / LightGBM.
- Deploy as a web app / REST API using Flask or Streamlit.

## 🙌 Acknowledgements

Submitted as part of the **AICTE | IBM SkillsBuild Data Analytics with AI
Academic Internship Program**, conducted by **BharatCares**.
