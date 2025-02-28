import pandas as pd, numpy as np, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import kagglehub
import os
import kagglehub

#Data Loading 

base_path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
file_name = "predictive_maintenance.csv"
file_path = os.path.join(base_path,file_name)

data = pd.read_csv(file_path)

print(data.head(5))
print(data.info())
print(data.describe())

#Data Cleaning

print(data.isnull().sum())

duplicates = data.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

if duplicates > 0:
    data.drop_duplicates(inplace=True)

#EDA

plt.figure(figsize=(5,4))
sns.countplot(x='Target',data=data)
plt.title('Distribution of Machine Failure (Target)')
plt.show()

data['product_quality'] = data['Product ID'].str[0]


plt.figure(figsize=(5,4))
sns.countplot(x='product_quality',data=data)
plt.title('Distribution of Product Quality(L,M,H)')
plt.show()

numeric_cols = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']

plt.figure(figsize=(20,18))
corr_matrix=data[numeric_cols].corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

sns.pairplot(data=data,vars=numeric_cols,hue='Target',corner=True, plot_kws={'alpha':0.5})
plt.suptitle('Pairplot of Numeric Featured by Machine Failure',y=1.02)
plt.show()

#Feature Engineering and Encoding
quality_map = {'L':0,'M':1,'H':2}
data['quality_encoded']=data['product_quality'].map(quality_map)


x = data[['Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'quality_encoded']]

y = data['Target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#Model Building

model = models.Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(16,activation='relu'),
    layers.Dense(8,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

#Train the model

history = model.fit(
    x_train,y_train,
    epochs=15, batch_size=32,
    validation_split=0.2,
    verbose=1
)

#Evaluate the model

y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs>0.5).astype(int)

print(confusion_matrix(y_test,y_pred))

classification_report(y_test,y_pred, zero_division=1)

print(history.history.keys())
plt.figure(figsize=(10,10))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results = model.evaluate(x_test,y_test,verbose=0)
loss=results[0]
accuracy=results[1]

print(f"Loss:{loss}, Accuracy: {accuracy}")