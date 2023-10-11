import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


df = pd.read_csv('SPY.csv')
df['Target'] = np.where(df['Close'] > df['Open'], 'green', 'red')

df['Yesterday_Close'] = df['Close'].shift(1)
df['Yesterday_Vol'] = df['Volume'].shift(1)
df['Yesterday_Pct_Change'] = abs(((df['Close'].shift(1) - df['Open'].shift(1)) / df['Open'].shift(1)) * 100)
df['Yesterday_Green'] = (df['Target'].shift(1) == 'green').astype(int)

df = df.dropna() #needed due to shift

X = df[['Volume', 'Yesterday_Green','Yesterday_Pct_Change']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

feature_names = X.columns.tolist()
class_names = model.classes_.tolist()

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()