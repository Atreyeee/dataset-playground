# DATASET_PLAYGROUND — Dataset Playground

Upload any CSV file and explore it visually: charts, statistics, and basic ML models — no coding needed.

---

## 🚀 How to Run (VS Code / Terminal)

### Step 1: Open Project Folder

Open VS Code and make sure you are inside the folder that contains:

* app.py
* requirements.txt
* utils/

---

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

---

### Step 3: Activate Environment

#### Windows:

```bash
venv\Scripts\activate
```

#### Mac / Linux:

```bash
source venv/bin/activate
```

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5: Run the App

```bash
streamlit run app.py
```

---

## 🌐 Output

* A local URL will appear in the terminal (like [http://localhost:8501](http://localhost:8501))
* The app will open automatically in your browser

---

## 📁 Project Structure

```
DATASET_PLAYGROUND/
│
├── app.py                ← Main Streamlit app
├── requirements.txt      ← Dependencies
├── README.md             ← Documentation
│
├── dataset/              ← Add your CSV files here (optional)
│
└── utils/
    ├── __init__.py
    ├── load_data.py
    ├── summary.py
    ├── visualization.py
    └── ml_model.py
```

---

## ✨ Features

* 📊 Overview

  * Upload CSV file
  * View dataset shape, columns, missing values

* 📋 Data Table

  * Browse and inspect full dataset

* 📈 Visualization

  * Histogram, scatter plot, correlation heatmap, box plot

* 🤖 ML Model

  * Train simple ML models
  * View accuracy / performance metrics

---

## ⚠️ Important Notes

* Make sure you run commands inside the correct folder
* Folder name must be exactly:
  utils/
* File names must match exactly:
  load_data.py
  ml_model.py
* Python is case-sensitive

---

## 🧠 Troubleshooting

* ModuleNotFoundError
  → Check folder name (utils) and file names

* requirements.txt not found
  → You are in the wrong directory → use cd DATASET_PLAYGROUND

* App not opening
  → Copy the URL from terminal and paste in browser

---

## 🚀 Future Improvements

* Add sample datasets
* Improve UI design
* Deploy online (Streamlit Cloud / Render)
