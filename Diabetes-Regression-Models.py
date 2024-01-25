#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd  # Import pandas module

# Load data
diabetes = load_diabetes()


X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial features
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_scaled)
X_test_poly = poly_features.transform(X_test_scaled)

# Models and predictions
models = {
    "Linear Regression": LinearRegression(),
    "RANSAC with LR": RANSACRegressor(estimator=LinearRegression()),
    "Ridge Regression": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(),
    "Polynomial Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(),
}

# Train and evaluate each model
r2_scores = {}
for name, model in models.items():
    if name == "Polynomial Regression":
        model.fit(X_train_poly, y_train)
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    r2_scores[name] = {"train": r2_score(y_train, y_pred_train), "test": r2_score(y_test, y_pred_test)}

#Print Results
print("R2 Scores:")
for name, scores in r2_scores.items():
    print(f"{name:25s}: Train R2: {scores['train']:.4f}, Test R2: {scores['test']:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:




