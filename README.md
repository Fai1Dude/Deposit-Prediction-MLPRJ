# 📈 Term-Deposit Prediction  
Machine-Learning mini-project (UCI “Bank Marketing” dataset)

This repo contains everything you need to reproduce a small end-to-end ML
pipeline that predicts whether a customer of a Portuguese bank will subscribe
to a **term-deposit** after a phone campaign.

| File | Role |
|------|------|
| `bank.csv` | Raw dataset (45 211 rows × 17 predictors + target) |
| `termProject-checkpoint.ipynb` | Jupyter notebook → data cleaning, EDA, model training & evaluation |

---

## 🔍 Dataset at a glance

* Source  : [UCI Machine-Learning Repository → Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
* Task   : binary classification (`y` = *yes* / *no*)  
* Domains  : socio-economic data, call statistics, campaign history  
* Format  : semicolon-separated CSV (**`;`**) with a header row added in the notebook

| Group | Example columns |
|-------|-----------------|
| Client profile | `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan` |
| Call details   | `contact`, `day`, `month`, `duration` |
| Campaign stats | `campaign`, `previous`, `poutcome` |
| **Target**     | `y` |

---

## 🗒️  Notebook workflow

1. **Load & clean** – add headers, cast numeric fields, replace `unknown`
   categories, one-hot encode categoricals.  
2. **Exploratory Data Analysis** – bar charts, correlation heat-map, box-plots.  
3. **Split** – stratified 70 / 30 train-test.  
4. **Model zoo** – Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting (optionally XGBoost).  
5. **Evaluation** – accuracy, precision/recall, ROC-AUC, confusion matrix.  
6. **Feature importance** – shows `duration`, `poutcome`, `contact` as top drivers.  
7. **Save** – serialises the best model to `model.pkl` with `joblib`.

*(Feel free to add more models or hyper-parameter searches; the notebook is fully editable.)*

---

## ⚙️  Quick start

```bash
# 1 ▪ clone the repo
git clone https://github.com/<your-user>/bank-term-deposit.git
cd bank-term-deposit

# 2 ▪ create & activate a virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3 ▪ install requirements
pip install notebook pandas scikit-learn matplotlib seaborn imbalanced-learn joblib

# 4 ▪ launch Jupyter
jupyter notebook termProject-checkpoint.ipynb
```

# 📊 Baseline results (course run)
| Model                | Accuracy | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | 88 %     | 0.79    |
| Decision Tree        | 90 %     | 0.82    |
| Random Forest        | **92 %** | **0.87**|
| Gradient Boosting    | 91 %     | 0.85    |

(Random Forest was chosen as the production model and saved to model.pkl.)
