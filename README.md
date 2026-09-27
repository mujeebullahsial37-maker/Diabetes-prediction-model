# Diabetes predictin model

A machine learning project that predicts whether a person is likely to have diabetes based on health, lifestyle, and demographic factors, using Logistic Regression.
## Dataset

The dataset contains the following features:

| Feature | Description |
|---|---|
| `Glucose` | Blood glucose level |
| `BMI` | Body Mass Index |
| `Insulin` | Insulin level |
| `Age` | Age of the individual |
| `Gender` | Male / Female |
| `Diet_Type` | Healthy / Moderate / Unhealthy |
| `Exercise_Frequency` | How often the person exercises |
| `Heredity` | Family history of diabetes (Yes/No) |
| `Smoking` | Smoking status (Yes/No) |
| `Alcohol` | Alcohol consumption level |
| `Stress_Score` | Numeric stress score |
| `Sleep_Hours` | Average hours of sleep |
| `Diabetes_Status` | Target variable — 1 (Diabetic) / 0 (Non-diabetic) |

##  Project Workflow

### 1. Data Cleaning
- Checked for missing values (`isnull().sum()`)
- Filled missing **numerical** columns (Glucose, BMI, Insulin, Age, Stress_Score, Sleep_Hours) using **mean imputation**
- Filled missing **categorical** columns (Gender, Diet_Type, Exercise_Frequency, Heredity, Smoking, Alcohol, Diabetes_Status) using **mode imputation**

### 2. Feature Scaling
- Applied `StandardScaler` to numerical columns (`Glucose`, `BMI`, `Insulin`, `Age`) to normalize their range

### 3. Train-Test Split
- Split the dataset into 80% training and 20% testing data using `train_test_split` (`random_state=0`)

### 4. Categorical Encoding
- Applied `OneHotEncoder` (with `drop="first"` to avoid the dummy variable trap) on categorical columns:
  `Gender`, `Diet_Type`, `Exercise_Frequency`, `Heredity`, `Smoking`, `Alcohol`
- Combined encoded categorical features with numerical features using `np.hstack`

### 5. Model Training
- Trained a **Logistic Regression** model (`sklearn.linear_model.LogisticRegression`) on the processed training data

### 6. Evaluation
- Predicted on the test set and evaluated using `accuracy_score`
- **Achieved Accuracy: 90.8%**

##  Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn (visualization)
- scikit-learn (`StandardScaler`, `OneHotEncoder`, `train_test_split`, `LogisticRegression`, `accuracy_score`)

##  Repository Structure

```
├── dataset.csv              # Raw dataset
├── Diabeties_pre.ipynb      # Jupyter notebook with full pipeline
├── Diabities_model.pkl      # Trained Logistic Regression model
└── README.md                # Project documentation
```

## How to Run

1. Clone this repository
   ```bash
   git clone <your-repo-url>
   ```
2. Install the required libraries
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `Diabeties_pre.ipynb` in Jupyter Notebook / VS Code and run the cells in order

##  Result

The Logistic Regression model achieved an accuracy of **90.8%** on the test set, indicating strong performance in predicting diabetes status based on the given health and lifestyle features.

## Future Improvements

- Try other models (Random Forest, XGBoost) for comparison
- Perform hyperparameter tuning
- Add cross-validation for more robust evaluation
- Handle class imbalance if present in `Diabetes_Status`


