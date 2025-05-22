# 📊 Anomaly Detection on UNSW-NB15 Dataset using Isolation Forest

This project demonstrates unsupervised anomaly detection on the **UNSW-NB15** intrusion detection dataset using the **Isolation Forest** algorithm. It includes preprocessing, model training, anomaly detection, visualization, and performance evaluation.

---

## 📁 Dataset

The dataset used is the **UNSW-NB15 training set**:  
`UNSW_NB15_training-set.csv`

> 🔗 You can download the dataset from the [official UNSW-NB15 website](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/).  
> 📥 Place the CSV file in the project directory before running the script.

---


## 🧠 Project Workflow

1. **Load and clean the dataset**
2. **Encode categorical features** (`proto`, `service`, `state`, `attack_cat`)
3. **Split the data** into train/test sets
4. **Train the Isolation Forest** model on the training data
5. **Predict anomalies** on the test data
6. **Visualize anomaly detection output**
7. **Evaluate model performance** using:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix
   - Classification Report

---

## 📉 Sample Output Metrics 
Accuracy : 0.7123  
Precision: 0.7685  
Recall : 0.6352 


> ⚠️ **Note:** Isolation Forest is an unsupervised model, so it does not train using labels. These metrics simply compare the model's anomaly predictions to the known attack labels.

---

## 📊 Visualization

The script generates a seaborn count plot of anomaly predictions:

- `-1` = Anomaly (likely attack)  
- `1` = Normal

This visualization helps in understanding how many data points were identified as anomalies versus normal.

---

## 👨‍💻 Author

- **Navya Saxena**
- GitHub: [@Nav-S24](https://github.com/Nav-S24)
