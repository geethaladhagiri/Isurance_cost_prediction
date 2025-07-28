# Hospital Insurance Charges Regression Project

This project predicts hospital insurance charges based on patient details like age, sex, BMI, children, smoking habits, and region.

## 📌 Dataset
The dataset used contains the following features:
- age
- sex
- bmi
- children
- smoker
- region
- charges (target)

## 🔧 Techniques Used
- Data preprocessing
- Encoding categorical features
- Exploratory Data Analysis (EDA)
- Linear Regression, Random Forest, XGBoost
- Evaluation metrics: RMSE, R² Score

## 📈 Results
- Achieved high R² with XGBoost regression.
- Best model: XGBoost with R² of ~0.88 on test data.

## 📂 Files
- `hospital_insurance_charges_regression_project.py`: Main code file

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python hospital_insurance_charges_regression_project.py
