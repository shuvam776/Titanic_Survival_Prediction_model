# 🚢 Titanic Survival Prediction using Machine Learning Pipeline

A production-style Machine Learning web application that predicts whether a passenger would have survived the Titanic disaster. The project demonstrates an end-to-end ML workflow, including data preprocessing, feature engineering, model training, and deployment using Flask with a modern Tailwind CSS interface.

---

## ✨ Features

* 🚢 Predicts passenger survival using a trained Logistic Regression model
* ⚙️ Complete Scikit-learn Pipeline for preprocessing and inference
* 🧩 Automatic feature engineering (Title extraction from passenger names)
* 📊 Displays prediction confidence using probability scores
* 🎨 Modern responsive UI built with Flask and Tailwind CSS
* 🔍 Interactive visualization of the Machine Learning Pipeline
* 📱 Mobile-friendly interface

---

## 🛠️ Tech Stack

* **Python**
* **Flask**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Tailwind CSS**
* **HTML5**

---

## 📊 Machine Learning Pipeline

```
Raw Passenger Data
        │
        ▼
Feature Engineering
(Name → Title Extraction)
        │
        ▼
Missing Value Handling
        │
        ▼
One-Hot Encoding
        │
        ▼
Feature Scaling
        │
        ▼
Logistic Regression
        │
        ▼
Survival Prediction
```

---

## 📂 Project Structure

```
TitanicApp/
│
├── templates/
│   ├── index.html
│   └── pipeline.html
│
├── static/
│
├── app.py
├── model_pipe.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/Titanic_Survival_Prediction_Model.git
```

### Navigate to the project

```bash
cd Titanic_Survival_Prediction_Model
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Flask application

```bash
python app.py
```

Visit:

```
http://127.0.0.1:5000
```

---

## 📈 Model Information

* **Algorithm:** Logistic Regression
* **Framework:** Scikit-learn Pipeline
* **Feature Engineering:** Passenger Title Extraction
* **Encoding:** One-Hot Encoding
* **Scaling:** StandardScaler
* **Output:** Survival Probability & Prediction

---

## 🖥️ Application Preview

The web application allows users to:

* Enter passenger information
* View prediction confidence
* Visualize the preprocessing pipeline
* Receive an instant survival prediction

---

## 📚 Dataset

The project uses the famous **Titanic - Machine Learning from Disaster** dataset provided by Kaggle.

---

## 💡 Future Improvements

* Explainable AI (SHAP/LIME)
* Model comparison dashboard
* Multiple ML algorithms
* Docker deployment
* REST API endpoints
* User prediction history
* Cloud deployment

---

## 👨‍💻 Author

**Shuvam Satapathi**

AI Undergraduate @ NIT Rourkela

Feel free to connect or contribute to improve this project!
