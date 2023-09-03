# Li_BMI500_Code_Review
This is Li's Code for Code Review in BMI500

# Gene Expression Prediction using Linear Regression

## Overview
This project uses a linear regression model to predict gene expression levels based on clinical information.

---

## Requirements
- Python 3.9.6
- scikit-learn==1.3.0
- pandas==2.0.3
- matplotlib==3.7.2
- numpy==1.25.2

---

## Data Source
The data for this project was downloaded from the CGGA official website and is sourced from two files: `CGGA.mRNAseq_325_clinical` and `CGGA.mRNAseq_325.RSEM-genes.20200506`. These files contain clinical data and gene expression levels for different patients. The expression level of a specific gene, KLRB1, was chosen as the prediction target for this study. The dataset used in this project, `CGGA_325_Read`, was preprocessed to represent non-numeric clinical information as numbers.

---

## Features

- Data Preprocessing: Missing value imputation and standardization.
- Prediction using scikit-learn's Linear Regression model.
- Model evaluation using Mean Squared Error (MSE) as a metric.
- Visualization: Scatter plot to compare true labels with predicted values.

---

## How to Run

1. Clone the repository.
2. Install the required packages.
3. The CSV file is in the same directory.
4. Run `code_review.py`.

---

## Data Preprocessing

### Missing Values
For feature columns, missing values are filled using the mode of the column. For the label column (KLRB1), rows with missing values are directly deleted.

### Standardization
Features are standardized using `StandardScaler`.

---

## Model
Uses scikit-learn's `LinearRegression` class for modeling and prediction.

---

## Evaluation
The model's performance is evaluated using Mean Squared Error (MSE).

---

## Visualization
A scatter plot is created using matplotlib to visually compare the true labels and predictions.

