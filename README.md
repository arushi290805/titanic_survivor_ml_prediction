# 🚢 Titanic Survivor Prediction (ML Project)

This project uses machine learning to predict the survival of passengers on the Titanic using the popular [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). The goal is to train a classification model based on features such as age, sex, passenger class, etc.

---

## 📁 Dataset Used

- **Train Dataset**: `train.csv`
- **Test Dataset**: `test.csv`
- Dataset includes features like:
  - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, etc.

---

## 🧼 Data Preprocessing

- Dropped irrelevant or high-cardinality columns:
  - `Name`, `Ticket`, `Cabin`, `Embarked`
- Handled missing values by filling them with `0`
- Converted categorical values (e.g., `Sex`) to numeric:
  - `male` → `0`, `female` → `1`
- Feature engineering on the `Ticket` column attempted using a `split_ticket()` function

---

## 🤖 Model Used

- **Logistic Regression** from `sklearn.linear_model`
- Trained using:
  - `x_train = train.drop("Survived", axis=1)`
  - `y_train = train["Survived"]`
- Evaluation libraries:
  - `accuracy_score`, `classification_report`, `confusion_matrix`

---

## 📈 Accuracy

- The model was evaluated using `accuracy_score`
- **Local Accuracy**: ~**78%**
- **Kaggle Submission Score**: ⭐ **0.76555**
- **Leaderboard Rank**: 🏅 **#10560** (User: `Arushi`, First submission!)

---

## 📦 Output

- Prediction results (`Survived`) added to the test dataset
- Output saved as:
  ```python
  test_new.to_csv("C:/Users/alokk/Documents/titanic_predictions.csv")
  ```

---

## 🏆 Kaggle Leaderboard

| Rank | Username | Score   | Submission | Note                |
|------|----------|---------|------------|---------------------|
| 10560| Arushi   | 0.76555 | 1st        | 🎉 Your First Entry! |

---

## 📌 Future Improvements

- Try models like Random Forest, XGBoost, or SVM
- Tune hyperparameters using `GridSearchCV`
- Handle missing values more robustly
- Add cross-validation
- Use feature scaling and interaction features

---

## 📚 Libraries Used

- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `sklearn.linear_model`, `sklearn.metrics`, `sklearn.preprocessing`

---

## 🚀 How to Run

1. Clone this repository
2. Install dependencies (via `pip` or `conda`)
3. Launch `Kaggle Titanic.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells to preprocess data, train the model, and generate predictions

---

## 🙌 Acknowledgements

Thanks to [Kaggle](https://www.kaggle.com/c/titanic) for hosting the Titanic competition and providing a great dataset for beginners in machine learning.
