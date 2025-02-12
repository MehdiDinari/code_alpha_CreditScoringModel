# Credit Scoring Model

## 📌 Overview
This project implements a **Credit Scoring Model** using **Python** and **Machine Learning (Scikit-Learn, XGBoost, etc.)** to classify individuals based on their creditworthiness. The model categorizes individuals into four credit risk levels:

- 🟢 **Good** - Eligible for credit
- 🟡 **Mid** - Medium risk, additional checks required
- 🔴 **Bad** - High risk, stricter conditions apply
- ⚫ **Not Available** - Credit denied

## 🚀 Features
- **Preprocessing:** Handles missing values, categorical encoding, and feature scaling.
- **Machine Learning:** Implements various classification algorithms (Random Forest, XGBoost, Logistic Regression, etc.).
- **Evaluation Metrics:** Uses accuracy, precision, recall, F1-score, and ROC-AUC for assessment.
- **Hyperparameter Tuning:** Utilizes GridSearchCV for optimization.
- **Deployment Ready:** Can be integrated into a web service or API.


## 🔧 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/credit-scoring-model.git
   cd credit-scoring-model
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## 📊 Dataset
The model is trained on a structured dataset containing:
- **Demographic Data** (Age, Employment status, Marital status, etc.)
- **Financial Data** (Income, Debt-to-Income Ratio, Existing Loans, etc.)
- **Credit History** (Payment History, Credit Score, Defaults, etc.)

Example dataset (CSV format):
```csv
ID, Age, Income, Debt, Credit_Score, Loan_Amount, Status
1, 35, 50000, 20000, 750, 15000, Good
2, 28, 30000, 15000, 620, 10000, Mid
3, 40, 70000, 30000, 500, 25000, Bad
```

## 📈 Model Training
Train the model using the following script:
```sh
python model_training.py
```
This script performs data preprocessing, feature selection, and trains multiple classifiers. The best model is selected based on performance metrics.

## 🏆 Model Evaluation
Evaluate the trained model using:
```sh
python model_evaluation.py
```
Example output:
```plaintext
Accuracy: 85.2%
Precision: 82.5%
Recall: 79.8%
F1-Score: 81.1%
ROC-AUC: 88.4%
```

## 🌍 Deployment
You can deploy the trained model as an API using Flask or Django.
Run the API locally:
```sh
python app.py
```
Example API request:
```sh
curl -X POST http://127.0.0.1:5000/predict -d '{"Age": 30, "Income": 40000, "Credit_Score": 700}' -H "Content-Type: application/json"
```
Response:
```json
{"credit_status": "Good"}
```

## 📜 License
This project is licensed under the MIT License.

## 🤝 Contributing
Feel free to contribute by submitting issues or pull requests.

## 📬 Contact
For any questions, reach out via [email@example.com](mailto:treshlol202@gmail.com) or visit [LinkedIn](https://www.linkedin.com/in/mehdi-dinari-b0487a2a9/).
