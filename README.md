# Sales Forecasting & Demand Intelligence System

An end-to-end machine learning dashboard for retail sales forecasting, anomaly detection, and product demand segmentation — built with **Streamlit**, **XGBoost**, **Isolation Forest**, and **KMeans** on the Superstore retail dataset.

---

## Features

| Module | Description |
|---|---|
| **Sales Overview** | Revenue KPIs, monthly trends, regional & category breakdowns, and a region x category heatmap |
| **Forecast Explorer** | Recursive XGBoost time-series forecasting with configurable horizons and confidence intervals |
| **Anomaly Report** | Dual-method anomaly detection using Isolation Forest + Rolling Z-Score with a business context register |
| **Demand Segments** | KMeans clustering (k=4) on sub-category features with PCA visualisation and a full segment mapping matrix |

---

## Tech Stack

- **Dashboard** — Streamlit
- **Forecasting** — XGBoost (recursive multi-step)
- **Anomaly Detection** — Isolation Forest + Rolling Z-Score (scikit-learn)
- **Clustering** — KMeans + PCA (scikit-learn)
- **Visualisation** — Plotly
- **Data** — pandas, NumPy

---

## Getting Started

### 1. Clone the repository
```
git clone https://github.com/arka-coder/End-to-End-Sales-Forecasting-Demand-Intelligence-System.git
cd End-to-End-Sales-Forecasting-Demand-Intelligence-System
```

### 2. Create and activate a virtual environment
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the dashboard
```
python -m streamlit run app.py
```

The app opens at http://localhost:8501

---

## Dataset

The project uses the **Sample Superstore** dataset (train.csv) — a US retail dataset containing ~9,800 orders across 4 regions, 3 product categories, and 17 sub-categories (2015-2018).

> **Note:** train.csv is included in this repository. If you wish to exclude it, uncomment the *.csv line in .gitignore.

---

## Project Structure

```
.
+-- app.py                  # Main Streamlit dashboard
+-- analysis.ipynb          # Exploratory analysis & model development (Tasks 1-6)
+-- train.csv               # Source dataset
+-- requirements.txt        # Python dependencies
+-- .gitignore
```

---

## Author

**Arka Roy** — Sales Forecasting & Demand Intelligence System
