import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



df = pd.read_csv("data/Housing.csv")

print(df.head())       
print(df.shape)       
df.info()              
print(df.describe())   

print(df.isnull().sum())
print(df.duplicated().sum())

# Price distribution
sns.histplot(df["price"], kde=True)
plt.show()

# Area distribution
sns.histplot(df["area"], kde=True)
plt.show()

# Bedroom counts
sns.countplot(x="bedrooms", data=df)
plt.show()

# Area vs price
sns.scatterplot(x="area", y="price", data=df)
plt.show()

# Bedrooms vs price
sns.boxplot(x="bedrooms", y="price", data=df)
plt.show()

# Furnishing status vs price
sns.boxplot(x="furnishingstatus", y="price", data=df)
plt.show()

numerical_cols = [
    "price", "area", "bedrooms",
    "bathrooms", "stories", "parking"
]

for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()



print("EDA Conclusions")
print("1. The dataset contains 545 houses and 13 columns.")
print("2. There are no missing values.")
print("3. There are no duplicate rows.")
print("4. House prices and areas vary across houses.")
print("5. The price box plot shows some potential high-price outliers.")


print(df.dtypes)

X = df.drop("price", axis=1)
y = df["price"]

print(X.head())
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
print("Pipeline trained successfully!")


y_pred = model.predict(X_test)

print("Predicted prices:")
print(y_pred[:5])

print("Actual prices:")
print(y_test.head())


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Linear Regression: Actual vs Predicted Prices")

plt.tight_layout()
plt.show()



# Linear Regression Evaluation
lr_mae = mean_absolute_error(y_test, y_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
lr_r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)




from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Create the model
dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

# 2. Train the model
dt_model.fit(X_train, y_train)

# 3. Make predictions
dt_pred = dt_model.predict(X_test)

# 4. Evaluate the model
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print("Decision Tree Results")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)
print("R2:", dt_r2)



from sklearn.ensemble import RandomForestRegressor

# 1. Create the model
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# 2. Train the model
rf_model.fit(X_train, y_train)

# 3. Make predictions
rf_pred = rf_model.predict(X_test)

# 4. Evaluate the model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Results")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)


import matplotlib.pyplot as plt

# Create two graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Decision Tree graph
axes[0].scatter(y_test, dt_pred, alpha=0.7)

axes[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

axes[0].set_title("Decision Tree: Actual vs Predicted")
axes[0].set_xlabel("Actual House Prices")
axes[0].set_ylabel("Predicted House Prices")


# 2. Random Forest graph
axes[1].scatter(y_test, rf_pred, alpha=0.7)

axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

axes[1].set_title("Random Forest: Actual vs Predicted")
axes[1].set_xlabel("Actual House Prices")
axes[1].set_ylabel("Predicted House Prices")

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# 1. Create a comparison table
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [lr_mae, dt_mae, rf_mae],
    "RMSE": [lr_rmse, dt_rmse, rf_rmse],
    "R2 Score": [lr_r2, dt_r2, rf_r2]
})

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

# 2. Plot MAE and RMSE
results.plot(
    x="Model",
    y=["MAE", "RMSE"],
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Comparison: MAE and RMSE")
plt.ylabel("Error in House Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Plot R2 Score
results.plot(
    x="Model",
    y="R2 Score",
    kind="bar",
    figsize=(8, 5),
    legend=False
)

plt.title("Model Comparison: R2 Score")
plt.ylabel("R2 Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np

# Create the Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Settings we want to test
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5]
}

# Try different combinations and select the best
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best settings
print("Best Settings:", grid_search.best_params_)
print("Best Cross-Validation MAE:", -grid_search.best_score_)

# Evaluate the tuned model on the test data
tuned_rf = grid_search.best_estimator_
tuned_pred = tuned_rf.predict(X_test)

tuned_mae = mean_absolute_error(y_test, tuned_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
tuned_r2 = r2_score(y_test, tuned_pred)

print("\nTuned Random Forest Results")
print("MAE:", tuned_mae)
print("RMSE:", tuned_rmse)
print("R2:", tuned_r2)

import joblib

joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")