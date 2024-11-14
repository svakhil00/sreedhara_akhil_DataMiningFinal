# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

# Calculates all Metrics
def evaluatePerformance(yTest, yPredictions, yProbabilities=None):
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(yTest, yPredictions).ravel()
    # Positive
    p = tp + fn
    # Negative
    n = tn + fp
    # True Positive Rate
    tpr = tp / p
    # True Negative rate
    tnr = tn / n
    # False Positive Rate
    fpr = fp / n
    # False Negative Rate
    fnr = fn / p
    # Recall
    r = tp / p
    # Precision
    precision = tp / (tp + fp)
    # F1 Measure
    f1 = 2 * (precision * r) / (precision + r)
    # Accuracy
    acc = (tp + tn) / (p + n)
    # Error Rate
    e = (fp + fn) / (p + n)
    # Specificity
    spc = tn / (fp + tn)
    # Negative Predictive Value
    npv = tn / (tn + fn)
    # False Discovery Rate
    fdr = fp / (fp + tp)
    # Balanced Accuracy
    bacc = .5 * (tp / (tp + fn) + tn / (tn + fp))
    # True Skill Statistics
    tss = tp / (tp + fn) - fp / (fp + tn)
    # Heidke Skill Score
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    # Brier Score
    bs = -123456789
    if yProbabilities is not None:
        bs = np.mean((yTest - yProbabilities) ** 2)
    else:
        bs = np.mean((yTest - yPredictions) ** 2)
    # Brier Skill Score (The formula in the notes forgot to do "1 -" in the beginning)
    bss = 1 - (bs / np.mean((yTest - np.mean(yTest)) ** 2))

    return {
        'True Positive': tp,
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive Rate': tpr,
        'True Negative Rate': tnr,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'Recall': r,
        'Precision': precision,
        'F1-Score': f1,
        'Accuracy': acc,
        'Error Rate': e,
        'Specificity': spc,
        'Negative Predictive Value': npv,
        'False Discovery Rate': fdr,
        'Balanced Accuracy': bacc,
        'True Skill Statistics': tss,
        'Heidke Skill Score': hss,
        'Brier Score': bs,
        'Brier Skill Score': bss
    }

# Prints Metrics in Tabular Formant
def printMetrics(performance, modelName):
    df = pd.DataFrame(performance, index=[0])
    print(f'\n{modelName}')
    print(df.T.to_string(header=False))

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[:100, [0, 1, 2, 3]].values

# Plot data
plt.scatter(X[:50, 0], X[:50, 2],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 2],
            color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# K-Fold 10 times
kf = KFold(n_splits=10, shuffle=True, random_state=42)

rfMetrics: list[dict] = []
svcMetrics: list[dict] = []
lstmMetrics: list[dict] = []

for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Splitting the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalize the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for LSTM (samples, timesteps, features)
    T = 1  # Set timesteps to 1 for basic LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], T, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], T, X_test_scaled.shape[1]))

    # Build the LSTM model
    lstm = Sequential()
    lstm.add(Input(shape=(T, X_train_scaled.shape[1])))
    lstm.add(LSTM(units=50, return_sequences=False))
    lstm.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=1)

    # Predict with LSTM
    lstmProbabilities = lstm.predict(X_test_lstm)
    lstmPredictions = (lstmProbabilities > 0.5).astype(int)

    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rfPredictions = rf.predict(X_test)

    # Support Vector Machine Classifier
    svc = LinearSVC(max_iter=10000)
    svc.fit(X_train, y_train)
    svcPredictions = svc.predict(X_test)

    # Calculate Performances
    performance = evaluatePerformance(y_test, rfPredictions)
    printMetrics(performance, 'Random Forest Metrics:')
    rfMetrics.append(performance)

    performance = evaluatePerformance(y_test, svcPredictions)
    printMetrics(performance, 'Support Vector Machine Metrics:')
    svcMetrics.append(performance)

    performance = evaluatePerformance(y_test, lstmPredictions, lstmProbabilities)
    printMetrics(performance, 'Long Short Term Memory Metrics:')
    lstmMetrics.append(performance)

# Overall Performances
printMetrics(pd.DataFrame(rfMetrics).mean().to_dict(), 'Overall Random Forest Metrics:')
printMetrics(pd.DataFrame(svcMetrics).mean().to_dict(), 'Overall Support Vector Machine Metrics:')
printMetrics(pd.DataFrame(lstmMetrics).mean().to_dict(), 'Overall Long Short Term Memory Metrics:')