import numpy as np
import pandas as pd
import math
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
st.title('Dieseas Prediction App')
st.info('This app helps you predict the kind of illness you have')
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('C:/Users/Admin/OneDrive/Documents/Python Scripts/Training.csv')
    df
    


# print(df.head())

df['labels'] = df.iloc[:,-2]
x = np.array(df.drop(columns=['prognosis', 'labels', 'Unnamed: 133'], axis=1))

label_encoder = preprocessing.LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['labels'])
# df['labels'].unique()
y = np.array(df["labels"])
# print(df['labels'].unique())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# print(x_test,y_test)
classifier = svm.SVR()
classifier.fit(x_train,y_train)
Accuracy = classifier.score(x_test,y_test)
# print(Accuracy)
col_names = list(df.columns)
# print(col_names)
col_names = col_names[:-3]
# print(col_names)
patient_sympt = []


# for col in col_names:
#     sympt = input(f'Do you have {col} (yes/no): ')  # Asking for the symptom
#     if sympt.lower() == 'yes':
#         patient_sympt.append(1)  # 1 for "yes"
#     else:
#         patient_sympt.append(0)  # 0 for "no"
patient_sympt = np.array(patient_sympt)
patient_sympt = patient_sympt.reshape(1,-1)
prediction = classifier.predict(patient_sympt)
label = label_encoder.classes_[int(prediction)]
print(f'You have {Accuracy*100:.2f}% chance of having {label}')
