# DDoS Attack Detection and Classification

A Machine Learning-based DDoS Attack Detection and Classification system that detects and classifies network attacks using multiple machine learning and deep learning models. The project includes dataset preprocessing, model training, performance evaluation, attack prediction, and real-time network traffic detection through an interactive graphical user interface.

---

## Features

- Dataset upload and preprocessing
- Feature selection and normalization
- Multiple Machine Learning and Deep Learning models
- Performance evaluation using standard metrics
- Model comparison dashboard
- CSV-based attack prediction
- Real-time DDoS attack detection
- Threat severity scoring
- Live attack timeline visualization
- Session logging and attack history
- Interactive GUI built with Tkinter

---

## Attack Classes

- BENIGN
- DrDoS_DNS
- DrDoS_LDAP
- DrDoS_MSSQL
- DrDoS_NTP
- DrDoS_NetBIOS
- DrDoS_SNMP
- DrDoS_SSDP
- DrDoS_UDP
- Syn
- UDP_LAG

---

## Machine Learning Models

- Naive Bayes
- Random Forest
- Support Vector Machine (Linear SVM)
- XGBoost
- Voting Ensemble
- Deep Neural Network (DNN)
- Long Short-Term Memory (LSTM)

---

## Technologies Used

- Python
- Tkinter
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- Scapy

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- K-Fold Cross Validation
- SHAP Feature Analysis

---

## Project Structure

```
DDOS-Attack-Detection-and-Classification
│
├── Dataset/
├── testData/
├── Main.py
├── requirements.txt
├── run.bat
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mattupalli-sneha/DDOS-Attack-Detection-and-Classification.git
```

Navigate to the project folder:

```bash
cd DDOS-Attack-Detection-and-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python Main.py
```

or

```bash
run.bat
```

---

## Workflow

1. Upload the dataset.
2. Preprocess the dataset.
3. Train one or more machine learning models.
4. Compare model performance.
5. Predict attacks using a CSV file.
6. Perform real-time DDoS detection.
7. Analyze results through charts and logs.

---

## Future Enhancements

- Web-based dashboard
- Cloud deployment
- Additional attack categories
- Enhanced real-time monitoring
- Advanced deep learning architectures

---

## License

This project is intended for educational and research purposes.
