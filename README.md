# 📊  Dataset Playground

An interactive, no-code **Data Analysis & Machine Learning platform** built with Streamlit.

DataLens allows users to **upload, explore, clean, visualize, and train ML models** on datasets — all from a single, intuitive interface.

Whether you're a beginner exploring datasets or a developer prototyping quickly, DataLens helps you move from **raw data → insights → models** in minutes.

---

## ✨ Key Features

### 📥 Data Upload & Overview

* Upload CSV files (up to 200MB)
* Load built-in sample dataset (Titanic)
* Instant dataset summary:

  * Rows & Columns
  * Missing Values
  * Numeric Columns
* Auto-generated insights:

  * Missing value warnings
  * Correlation hints
  * Data distribution checks

---

### 🧹 Data Cleaning & Management

* Interactive data table view
* Filter and search data easily
* One-click cleaning:

  * Remove duplicates
  * Fill missing values (Mean / Mode)
* Export filtered dataset

---

### 📈 Dynamic Visualization

Create multiple charts without writing code:

* Distribution plots
* Correlation heatmaps
* Scatter plots
* Box / Violin plots
* Bar charts
* Line plots
* Pie charts
* Histograms
* Pairplot

---

### 🤖 Machine Learning (Auto ML)

* Select target column
* Automatically detects:

  * Classification
  * Regression
* Trains multiple models:

  * Logistic / Linear Regression
  * Random Forest
  * Decision Tree
* Compares model performance
* Displays:

  * Accuracy / R² score
  * Feature importance
  * Confusion matrix / residual plots

---

## 🛠️ Tech Stack

* **Frontend & Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn

---

## 🚀 Getting Started

### 🔹 Prerequisites

* Python 3.8 or higher
  Download: https://www.python.org/downloads/

---

### 🔹 Installation

#### 1. Clone Repository

```bash
git clone https://github.com/your-username/datalens-dataset-playground.git
cd datalens-dataset-playground
```

---

#### 2. Create Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

#### 4. Run the App

```bash
streamlit run app.py
```

👉 App will open at:
http://localhost:8501

---

## 📂 Project Structure

```
Dataset-Playground/
│
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── README.md # Documentation
│ ├── utils/ # Core logic modules
│ ├── __init__.py
│ ├── load_data.py # CSV loading logic
│ ├── summary.py # Dataset summary & insights
│ ├── visualization.py # Plotting functions
│ └── ml_model.py # Machine learning logic
└── Datasets/
# Sample datasets (Excel/CSV files)
├── dataset1.xlsx
├── dataset2.xlsx
```

---

## 💡 Advantages

* **No-Code Interface**
  Perform EDA and ML without writing code

* **Privacy First**
  Runs locally — no data leaves your machine

* **Fast Prototyping**
  Quickly explore and model any dataset

---

## ⚠️ Limitations

* Only supports `.csv` files
* Basic missing value handling (Mean/Mode only)
* No hyperparameter tuning for ML models
* Large datasets (~200MB) may slow performance

---

## 🤝 Contributing

Contributions are welcome!

### Steps:

```bash
# Fork the repository

# Clone your fork
git clone https://github.com/your-username/datalens-dataset-playground.git

# Create a new branch
git checkout -b feature/your-feature

# Make changes
git commit -m "Add new feature"

# Push changes
git push origin feature/your-feature
```

Then open a Pull Request.

---

## 💡 Contribution Ideas

* Add support for `.xlsx` and `.json`
* Improve data cleaning (KNN, forward fill)
* Add model download (.pkl)
* Improve UI/UX styling
* Add hyperparameter tuning

---

## 🔥 Final Note

DataLens is designed to bridge the gap between **data exploration and machine learning**, making powerful tools accessible through a simple interface.

---

⭐ If you like this project, consider giving it a star!
