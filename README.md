Fraud Detection using PyOD AutoEncoder

This project uses an AutoEncoder model from the PyOD (Python Outlier Detection) library to detect fraudulent credit card transactions. The AutoEncoder is trained on non-fraudulent data to establish a baseline for 'normal' behavior, classifying high reconstruction error as an anomaly (fraud).

1. Prerequisites

You must have Python 3.10+ installed.

2. Dataset

The dataset used is the publicly available Credit Card Fraud Detection dataset from Kaggle.

Source: Kaggle: Credit Card Fraud Detection

NOTE: Ensure you download the creditcard.csv file and place it in the data/ directory before running the script.

3. How to Run

Step 1: Install Dependencies

Open your terminal or command prompt and run the following command to install all required libraries:

pip install pyod pandas numpy scikit-learn matplotlib


Step 2: Folder Structure

Ensure your project structure matches the following (create the folders if they don't exist):

fraud_detection_pyod/
│
├── data/
│   └── creditcard.csv  <-- PLACE THE DOWNLOADED FILE HERE
│
├── src/
│   ├── autoencoder_fraud.py
│   └── manifest.txt
│
├── screenshots/
│   └── output.png
│
└── README.md


Step 3: Execute the Script

Navigate to the src/ directory and run the Python script:

python3 src/autoencoder_fraud.py


4. Output

The script will print the training progress, the resulting Confusion Matrix, and the Classification Report to the console.

A screenshot of this terminal output should be saved in the screenshots/output.png file for submission.
