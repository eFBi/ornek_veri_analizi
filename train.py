# -*- coding: utf-8 -*-
"""

Veri setimizde:
* Çalışan memnuniyet oranı,
 (satisfaction_level) (0-1aralığında)
* Son değerlendirme (last_evaluation) (0-1 aralığında)
* Proje sayısı (number_project)
* Ortalama aylık çalışma süresi (average_monthly_hours)
* Şirkette geçirilen yıl (time_spent_company)
* İş kazası geçirilip geçirilmediği (work_accident)
* Son 5 yılda promosyon alıp almadığı (promotion_last_5_years)
* Departman(sales)
* Maaş (salary) - (low, medium or high)
* Çalışanın işten ayrılıp ayrılmadığı (left)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd .read_csv("HR.csv")

df.head(3)

df.info()

df.describe()

df.isnull().sum()

df.corr()["left"].sort_values()

df.head(5)

kolerasyon = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','left','promotion_last_5years']]
              
       
plt.figure(figsize=(20, 8))
sbn.heatmap(kolerasyon.corr(),annot = True,  cmap='RdYlGn_r', mask=np.triu(np.ones_like(kolerasyon.corr())));
plt.title('Correlations between factors', fontsize=20, fontweight='bold', pad=20);

df.head(2)

plt.figure(figsize=(15,4),dpi=100)
plt.subplot(1,2,1)
sbn.distplot(df['satisfaction_level'])
plt.subplot(1,2,2)
sbn.distplot(df['Work_accident'])

plt.figure(figsize=(10,4),dpi=100)

sbn.distplot(df['time_spend_company'])

sbn.boxplot(x = df.satisfaction_level, color = 'blue')
plt.show()

sbn.boxplot(x=df.time_spend_company, color = 'yellow')
plt.show()

df[df['time_spend_company']>=8].count()

yeniDf = df[df['time_spend_company']<8].count()#outliar olarak düşündüğüm verileri sildim.

plt.figure(figsize=(12,5),dpi=100)
plt.title('Sales Özelliğine Değerlerine Göre Left')
sbn.countplot(x='sales', data=df, hue='left')
plt.show()

"""## Veriyi test/train olarak ikiye ayırma"""

from sklearn.model_selection import train_test_split

# Y = wX + b

# Y -> Label (Çıktı)
y = df["left"].values

# X -> Feature,Attribute (Özellik)
x = df[['satisfaction_level','time_spend_company','Work_accident']].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=10)

print(x_train.shape,x_test.shape,y_train.shape,x_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

"""## Yapay sinir ağları modeli

"""

import tensorflow as tsf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

np.random.seed(0)

model = Sequential()
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

model.fit(x=x_train,y=y_train,epochs=250,validation_data=(x_test,y_test),verbose=1)

model_loss = pd.DataFrame(model.history.history)

figure = model_loss.plot()
plt.show()

tahminlerimiz = model.predict_classes(x_test)

model.evaluate(x_train,y_train)

"""## modeli değerlendir"""

from sklearn.metrics import classification_report , confusion_matrix, accuracy_score

print(classification_report(y_test,tahminlerimiz))

print(confusion_matrix(y_test,tahminlerimiz))

acc_ann = accuracy_score(y_test,tahminlerimiz)
print('Accuracy = ', acc_ann)

"""## Karar ağacı yöntemi """

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for leaves in range(2,11):
    classifier = DecisionTreeClassifier(max_leaf_nodes = leaves, random_state=0, criterion='entropy')#
    classifier.fit(x_train, y_train)
    y_tahmin = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_tahmin))
plt.plot(list(range(2,11)), list1)
plt.show()

classifier = DecisionTreeClassifier(max_leaf_nodes = 3, random_state=0, criterion='entropy')
classifier.fit(x_train, y_train)

y_tahmin = classifier.predict(x_test)
print(y_tahmin)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_tahmin)
print(cm)

acc_dt = accuracy_score(y_test,y_tahmin)
print('Accuracy = ', acc_dt)

"""## Modellerin karşılaştırılması"""

modeller = ['ANN','Decision Tree']
accuracy_list = [acc_ann*100,acc_dt*100]
accuracy_list

plt.figure(figsize=(8,3),dpi=100)
sbn.barplot(x=accuracy_list,y=modeller,palette='Blues')
plt.xlabel("Accuracy")
plt.ylabel("Modeller")
plt.title('Modellerin Doğruluk Karşılaştırılması')
plt.show()

