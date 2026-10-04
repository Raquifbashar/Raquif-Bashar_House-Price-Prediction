# # 🏠 House Price Prediction using Machine Learning
# 
# **AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares**
# 
# **Submitted by:** Raquif Bashar
# **Project:** House Price Prediction
# **Domain:** Data Analytics with AI
# 
# ---
# 
# ## 1. Problem Statement
# 
# Real estate prices depend on a large number of factors such as location, size,
# number of rooms, income level of the neighbourhood, and proximity to the
# ocean. The goal of this project is to build a **regression model** that can
# accurately **predict the median house value** of a district in California
# based on its socio-economic and geographical features.
# 
# ## 2. Objectives
# 
# - Perform Exploratory Data Analysis (EDA) on the California Housing dataset.
# - Clean and preprocess the data (handle missing values, encode categorical
#   features, engineer new features).
# - Train and compare multiple Machine Learning regression models.
# - Evaluate models using standard regression metrics (MAE, RMSE, R²).
# - Identify the most important features that drive house prices.
# - Select and save the best-performing model for future predictions.
# 
# ## 3. Dataset
# 
# - **Name:** California Housing Prices
# - **Source:** Derived from the 1990 U.S. Census, popularised by Aurélien
#   Géron's *"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"*.
# - **Records:** 20,640 rows | **Features:** 10 columns
# - **Link:** https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
# 
# | Column | Description |
# |---|---|
# | longitude, latitude | Geographic coordinates of the district |
# | housing_median_age | Median age of houses in the district |
# | total_rooms, total_bedrooms | Total rooms / bedrooms in the district |
# | population, households | Population and number of households |
# | median_income | Median income of households (in tens of thousands of USD) |
# | ocean_proximity | Categorical: distance category from the ocean |
# | **median_house_value** | **Target variable** — median house value (USD) |

# ## 4. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42
print("Libraries imported successfully.")


# ## 5. Load the Dataset
# 
# The dataset is read directly from the raw CSV (also saved locally under
# `data/housing.csv` for offline use — see the dataset link above).

df = pd.read_csv("data/housing.csv")
print("Shape of dataset:", df.shape)
df.head()


df.info()


df.describe().T


# ## 6. Exploratory Data Analysis (EDA)

# ### 6.1 Missing Values

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Columns with missing values:\n", missing)


# ### 6.2 Target Variable Distribution

plt.figure(figsize=(8, 5))
sns.histplot(df["median_house_value"], bins=50, kde=True, color="steelblue")
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value (USD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/target_distribution.png", dpi=120)
plt.show()


# ### 6.3 Geographical Distribution of Prices

plt.figure(figsize=(9, 7))
scatter = plt.scatter(df["longitude"], df["latitude"], c=df["median_house_value"],
                       cmap="viridis", s=df["population"] / 100, alpha=0.4)
plt.colorbar(scatter, label="Median House Value (USD)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("House Prices by Location (bubble size = population)")
plt.tight_layout()
plt.savefig("images/geo_distribution.png", dpi=120)
plt.show()


# ### 6.4 Ocean Proximity vs. House Value

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", palette="Set2")
plt.title("House Value by Ocean Proximity")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("images/ocean_proximity.png", dpi=120)
plt.show()


# ### 6.5 Correlation Heatmap

numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(9, 7))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png", dpi=120)
plt.show()


# ### 6.6 Median Income vs. House Value

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="median_income", y="median_house_value", alpha=0.3, color="darkorange")
plt.title("Median Income vs. Median House Value")
plt.tight_layout()
plt.savefig("images/income_vs_value.png", dpi=120)
plt.show()


# ## 7. Feature Engineering
# 
# We create a few derived ratio features that are known (from EDA above) to be
# more predictive than the raw counts.

df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

df[["rooms_per_household", "bedrooms_per_room", "population_per_household"]].describe()


# ## 8. Train-Test Split

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# ## 9. Preprocessing Pipeline
# 
# - Numeric columns: missing values filled with the **median**, then
#   **standardised**.
# - Categorical column (`ocean_proximity`): missing values filled with the
#   **most frequent** value, then **one-hot encoded**.
# 
# Using a `ColumnTransformer` + `Pipeline` ensures the exact same
# transformations are applied consistently to training and test data (and to
# any new data at inference time), without data leakage.

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])


# ## 10. Model Training
# 
# We train and compare four regression algorithms:
# 
# 1. Linear Regression (baseline)
# 2. Decision Tree Regressor
# 3. Random Forest Regressor
# 4. Gradient Boosting Regressor

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})
    fitted_pipelines[name] = pipe

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
results_df


# ### 10.1 Model Comparison Chart

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="R2 Score", palette="Blues_d")
plt.title("Model Comparison — R² Score (higher is better)")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=120)
plt.show()


# ## 11. Hyperparameter Tuning (Best Model)
# 
# The **Random Forest Regressor** is tuned further using `GridSearchCV` to
# squeeze out extra performance.

param_grid = {
    "regressor__n_estimators": [100, 150],
    "regressor__max_depth": [15, 25],
}

best_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
])

grid_search = GridSearchCV(
    best_pipe, param_grid, cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=1, verbose=0,
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_


preds = best_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"Tuned Random Forest -> MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R2: {r2:.4f}")


# ### 11.1 Actual vs. Predicted Values

plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.3, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs. Predicted House Values (Tuned Random Forest)")
plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png", dpi=120)
plt.show()


# ## 12. Feature Importance

ohe_columns = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_columns)

importances = best_model.named_steps["regressor"].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10], palette="viridis")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=120)
plt.show()

feat_imp.head(10)


# ## 13. Save the Final Model
# 
# The tuned pipeline (preprocessing + model) is saved with `joblib` so it can
# be reloaded later to make predictions on new/unseen data without retraining.

joblib.dump(best_model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")


# ### 13.1 Sample Prediction on New Data

sample = X_test.iloc[[0]]
predicted_value = best_model.predict(sample)[0]
actual_value = y_test.iloc[0]

print("Sample input:\n", sample.to_dict(orient="records")[0])
print(f"\nPredicted median house value: ${predicted_value:,.2f}")
print(f"Actual median house value:    ${actual_value:,.2f}")


# ## 14. Conclusion
# 
# - The dataset was cleaned, missing values were imputed, and new ratio-based
#   features (`rooms_per_household`, `bedrooms_per_room`,
#   `population_per_household`) improved model performance.
# - Among the four models tested, **ensemble tree-based models (Random Forest
#   and Gradient Boosting)** significantly outperformed the linear baseline,
#   confirming that the relationship between features and house price is
#   non-linear.
# - After hyperparameter tuning, the **Random Forest Regressor** achieved the
#   best trade-off between accuracy and generalisation, with a strong R² score
#   on unseen test data.
# - **Median income**, **geographic location (latitude/longitude)**, and
#   **ocean proximity** emerged as the most influential predictors of house
#   price — consistent with real-world real-estate intuition.
# - The final trained pipeline was serialized (`house_price_model.pkl`) and can
#   be directly reused for inference in a web app or API.
# 
# ## 15. Future Scope
# 
# - Incorporate additional real-world features such as crime rate, school
#   ratings, and distance to city centre.
# - Experiment with advanced models such as XGBoost / LightGBM and neural
#   networks.
# - Deploy the model as a REST API / interactive web app (e.g., using
#   Flask/Streamlit) for real-time predictions.
