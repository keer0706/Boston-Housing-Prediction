# House Price Prediction

## Overview

This project uses the Boston Housing dataset to build and compare machine learning regression models for predicting median house values.

The project covers data loading, exploratory data analysis (EDA), missing-value handling, correlation analysis, visualization, feature scaling, model training, and regression model evaluation.

## Dataset

The dataset is stored in `HousingData.csv`.

- Number of observations: 506
- Number of columns: 14
- Target variable: `MEDV`
- Problem type: Regression

### Features

| Feature | Description |
|---|---|
| `CRIM` | Per-capita crime rate |
| `ZN` | Proportion of residential land zoned for large lots |
| `INDUS` | Proportion of non-retail business acres |
| `CHAS` | Charles River dummy variable |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to employment centres |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Property-tax rate |
| `PTRATIO` | Pupil-teacher ratio |
| `B` | Demographic-related transformed feature |
| `LSTAT` | Percentage of lower-status population |
| `MEDV` | Median value of owner-occupied homes; target variable |

## Project Workflow

### 1. Data Loading and Inspection

The dataset is loaded using Pandas and examined using:

- `head()`
- `columns`
- `shape`
- `info()`
- `describe()`
- duplicate-value checks
- missing-value checks

### 2. Missing-Value Handling

Missing values were handled using different imputation strategies based on the feature distributions.

- Median imputation was used for `CRIM`, `ZN`, `CHAS`, and `LSTAT`.
- Mean imputation was used for `INDUS` and `AGE`.

### 3. Exploratory Data Analysis

The project includes:

- Feature boxplots
- Histograms for `RM`, `LSTAT`, and `PTRATIO`
- Correlation analysis
- Correlation heatmap

The correlation analysis showed that:

- `RM` has a strong positive correlation with `MEDV` (approximately `0.70`).
- `LSTAT` has a strong negative correlation with `MEDV` (approximately `-0.72`).
- `CHAS` and `DIS` have relatively weak individual correlations with `MEDV`.
- `RAD` and `TAX` have a strong correlation with each other (approximately `0.91`), indicating possible multicollinearity.

### 4. Train-Test Split

The data was divided into training and testing sets using an 80:20 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 5. Feature Scaling

`StandardScaler` was applied for:

- Linear Regression
- K-Nearest Neighbors (KNN)

The Decision Tree Regressor was trained on the original feature values because tree-based models do not require feature scaling.

### 6. Models Used

Three regression models were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. K-Neighbors Regressor

### 7. Evaluation Metrics

The models were evaluated using:

- **MAE (Mean Absolute Error):** lower is better
- **MSE (Mean Squared Error):** lower is better
- **RMSE (Root Mean Squared Error):** lower is better
- **R² Score:** higher is better

## Results

The results saved in the notebook are:

| Model | MAE | R² | MSE | RMSE |
|---|---:|---:|---:|---:|
| **Decision Tree Regressor** | **2.851** | **0.811** | **13.858** | **3.723** |
| K-Neighbors Regressor | 2.744 | 0.697 | 22.255 | 4.718 |
| Linear Regression | 3.154 | 0.659 | 25.028 | 5.003 |

### Best Model

The **Decision Tree Regressor** achieved the highest R² score and the lowest MSE and RMSE among the models evaluated in the notebook.

- R²: **0.811**
- RMSE: **3.723**
- MAE: **2.851**

This indicates that the Decision Tree Regressor provided the best overall predictive performance among the three evaluated models.

## Visualizations

The project generates and saves several visualizations, including:

- Feature boxplots
- RM distribution histogram
- LSTAT distribution histogram
- PTRATIO distribution histogram
- Correlation heatmap

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## How to Run

1. Open the `House_Price_Prediction.ipynb` notebook in Google Colab or Jupyter Notebook.
2. Upload `HousingData.csv` to the working directory.
3. Run the notebook cells from top to bottom.
4. The model comparison table will show the MAE, R², MSE, and RMSE values for each model.

## Project Structure

```text
House-Price-Prediction/
│
├── House_Price_Prediction.ipynb
├── HousingData.csv
├── README.md
│
└── visualizations/
    ├── heatmap.png
    ├── features_boxplot.png
    ├── RM_histogram.png
    ├── LSTAT_histogram.png
    └── PTRATIO_histogram.png
```

## Key Takeaway

The project demonstrates a complete basic machine learning regression workflow, from data preprocessing and exploratory analysis to model training and evaluation. Among the tested models, the Decision Tree Regressor performed best on the test set.
