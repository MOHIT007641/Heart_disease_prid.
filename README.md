![1](https://github.com/user-attachments/assets/fc753578-e3e2-4a23-be2f-cb611e8405c8)
![2](https://github.com/user-attachments/assets/a2661d48-1f73-4b20-b153-4804ffd0ab36)
![3](https://github.com/user-attachments/assets/be00ea6f-95ce-4d9a-adbd-6d984cd57df5)
![4](https://github.com/user-attachments/assets/aa9e7caa-3dec-41f5-bb51-03d36f7cc966)
![5](https://github.com/user-attachments/assets/f74c0f70-52de-4539-9b10-8eb51dd6ede7)
![6](https://github.com/user-attachments/assets/61183d48-71d3-4233-a9d9-86a18236f627)
![7](https://github.com/user-attachments/assets/8c65b366-7692-47ee-9d7f-e070bdc69f07)
# Heart_disease_prid.
Link to the code https://colab.research.google.com/drive/1G0Ov6X6KjcuSK-B6JiYZG8hhAORx_Iwm?usp=sharing

This project utilizes machine learning techniques to predict heart disease based on various health parameters. The dataset consists of multiple attributes related to cardiovascular health and is analyzed using Python.

Technologies Used

Python

NumPy

Pandas

Scikit-Learn (Logistic Regression, Model Evaluation)

Dataset

The dataset used in this project contains patient records with the following features:

Age

Sex

Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Cholesterol Level (chol)

Fasting Blood Sugar (fbs)

Resting ECG Results (restecg)

Maximum Heart Rate Achieved (thalach)

Exercise-Induced Angina (exang)

ST Depression Induced by Exercise (oldpeak)

Slope of the Peak Exercise ST Segment (slope)

Number of Major Vessels (ca)

Thalassemia (thal)

Target (0 = No Disease, 1 = Disease Present)

Installation

To run this project, install the required dependencies:

pip install numpy pandas scikit-learn

Usage

Load the dataset:

import pandas as pd
heart_data = pd.read_csv('heart_disease_data.csv')
print(heart_data.head())

Preprocess the data:

from sklearn.model_selection import train_test_split
X = heart_data.drop(columns=['target'], axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

Train the model:

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

Evaluate the model:

from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

Results

The model is trained using Logistic Regression and achieves a reasonable accuracy.

Further improvements can be made using other ML algorithms such as Decision Trees, Random Forest, or Neural Networks.



