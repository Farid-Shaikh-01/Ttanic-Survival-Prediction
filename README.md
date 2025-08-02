# 🚢 Titanic Survival Prediction using Machine Learning

🎯 Predicting who survived the Titanic tragedy using real passenger data and machine learning!

This project uses a clean pipeline with preprocessing, Random Forest Classifier, and hyperparameter tuning to build a reliable classification model. It’s a great real-world example of using **machine learning for binary classification**.

---

## 📊 Dataset Overview

I used the famous **Titanic dataset** from the `seaborn` library (similar to Kaggle's version).

**Features Used**:
- `pclass` - Passenger Class (1st, 2nd, 3rd)
- `sex` - Gender
- `age` - Age of passenger
- `fare` - Ticket fare
- `embarked` - Port of embarkation

🎯 **Target**: `survived` (0 = No, 1 = Yes)

---

## 🔧 Machine Learning Pipeline

✅ Built using **`scikit-learn` Pipelines**:
- **Numerical features**: Scaled using `StandardScaler`
- **Categorical features**: Encoded using `OneHotEncoder`
- Final model: **Random Forest Classifier**

🔍 **Tuned with GridSearchCV** for:
- `n_estimators`: [50, 100]
- `max_depth`: [3, 5, None]

---

## 🧠 Model Evaluation

✅ **Best Parameters**: Found using `GridSearchCV`  
📈 **Model Performance**:

| Metric          | Value    |
|-----------------|----------|
| **Accuracy**    | ✅ 0.8041 (80.41%) |
| **AUC Score**   | ✅ 0.8433 (84.33%) |
| **Precision (1)** | 0.89     |
| **Recall (0)**    | 0.94     |

📊 **Confusion Matrix Preview**:  
*(Taken from actual output)*

![Confusion Matrix](images/confusion_matrix.png)

> The model performs very well at predicting survivors and non-survivors, with good balance across metrics.

---

## 💾 Model Saving & Loading

The trained model is saved as a `.pkl` file for future use — no need to retrain!

```python
# Save the model
import joblib
joblib.dump(grid.best_estimator_, 'titanic_model.pkl')

# Load the model later
model = joblib.load('titanic_model.pkl')
print("Model loaded successfully!")
```
🧠 The saved model includes:

Preprocessing steps

GridSearch-tuned Random Forest

Full pipeline ready for deployment or testing!


📁 File: titanic_model.pkl


💻 Installation Instructions

pip install -r requirements.txt

👨‍💻 Author

Farid Shaikh
Machine Learning Learner | Turning Data into Decisions
🔗 [LinkedIn Profile (https://www.linkedin.com/in/farid-shaikh-937734338)]


🌟 If you like this project...

Give it a ⭐ on GitHub!
It motivates me to create more real-world ML projects.

#MachineLearning #Titanic #RandomForest #Python #GitHubProjects #FaridShaikh
