import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.impute import KNNImputer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
import keras 

connect = sqlite3.connect("Data-Heart-Disease/heart.db")
heart = pd.read_sql("SELECT * FROM heart" , connect)

print(heart.head())
print(heart.info())
print(heart.)
#  Exploratory Data Analysis


for col in heart.columns:
    print(f"{col} has {heart.nunique()} values")
    break

plt.figure(figsize=(10,8))
ax = sns.countplot(x=heart['target'] , palette='pastel')
for p in ax.patches:
    ax.text(p.get_x()+p.get_width()/2,p.get_height()+3, f'{p.get_height()}',ha="center")
plt.savefig("Heart-Disease.png" , bbox_inches='tight')
  
  
# Distribution of Categorical Variables

plt.figure(figsize=(15,5))
number = 1

for col in heart.columns:
    if heart[col].nunique() < 5:
        if number <= 6:
            plt.subplot(2,3,number)
            ax = sns.countplot(x=heart[col], palette='pastel')

            for p in ax.patches:
                ax.text(p.get_x()+p.get_width()/2,p.get_height()+3,f"{int(p.get_height())}",ha='center')

        number += 1

plt.suptitle("Distribution of Categorical Variables" , fontsize=40, y=1)
plt.tight_layout()
plt.savefig("Distribution-of-Categorical-Variables.png" , bbox_inches='tight')


# Distribution of Categorical Variables by Target

plt.figure(figsize=(15,5))
number = 1

for col in heart.columns:
    
    if heart[col].nunique() < 5:
        if number <= 6:
            plt.subplot(2,3,number)
            ax = sns.countplot(x=heart[col],hue=heart['target'] , palette='bright')


            for p in ax.patches:
                ax.text(p.get_x()+p.get_width()/2., p.get_height()+3,f"{int(p.get_height())}",ha='center')

        number += 1

plt.suptitle("Distribution of Categorical Variables by traget" , fontsize=40, y=1)
plt.tight_layout()
plt.savefig("Distribution-of-Categorical-Variables-by-Target.png" , bbox_inches='tight')

# Replace zeros in the 'Serum Cholesterol' column with numpy.nan
# because zero is not a valid value for cholesterol levels
# and should be considered as missing data instead.

heart['cholesterol'] = heart['cholesterol'].replace(0,np.nan)
heart['cholesterol'].isnull().sum()
knn_imputer = KNNImputer(n_neighbors=5)
heart = pd.DataFrame(knn_imputer.fit_transform(heart),columns=heart.columns)
heart['cholesterol'].isnull().sum()

for col in heart.columns:
    if col != 'oldpeak':
        heart[col] = heart[col].astype(int)

heart.to_sql('heart',connect,if_exists='replace' , index=False)

plt.figure(figsize=(20,5))
number = 1

for col in heart.columns:

    if heart[col].nunique() > 5:
        plt.subplot(2,3,number)
        sns.kdeplot(heart[col],fill=True,color='deepskyblue')

        number += 1

plt.suptitle("Distribution of Numerical Variables" , fontsize=40,y=1)
plt.tight_layout()

plt.savefig("Distribution-of-Numerical-Variables2.png" , bbox_inches='tight')          


plt.figure(figsize=(35,8))
number = 1

num_columns = []

for col in heart.columns:
    if heart[col].nunique() > 5:
        num_columns.append(col)

for col in num_columns:
    #Histogram
    ax = plt.subplot(len(num_columns) , 2 , number) 
    sns.histplot(heart[col],kde=True,color='darkorange')
    plt.xlabel(col)
    plt.grid(False)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    number += 1

    ax = plt.subplot(len(num_columns) , 2 , number)
    sns.boxplot(x=heart[col] , color='darkorange' , width=0.8,linewidth=1)   
    plt.xlabel(col)
    plt.grid(False) 

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    number += 1

plt.suptitle('Distribution of Numerical Variables', fontsize=40,y=1.02)
plt.tight_layout()
plt.savefig('Distribution-of-Numerical-Variables2.png',  bbox_inches='tight')

plt.figure(figsize=(20,5))
number = 1

for col in heart.columns:

    if heart[col].nunique() > 5:
        plt.subplot(2,3,number)
        sns.histplot(data=heart,x=col,hue='target',kde=True,palette='bright') 
        plt.xlabel(col)
        number += 1

plt.suptitle("Distribution of Numerical Variables by Target3" , fontsize=9.0,y=1)
plt.tight_layout()
plt.savefig("Distribution-of-Numerical-Variables-by-Target3.png" , bbox_inches='tight')    

plt.figure(figsize=(17,10))

sns.heatmap(heart.corr(),annot=True,cmap='coolwarm',linewidths=2,linecolor='lightgrey')

plt.suptitle("Correlation Matrix" , fontsize=40,y=1)
plt.savefig("Correlation-Matrix.png" , bbox_inches='tight')


x = heart.drop(['target'] , axis=1)
y = heart['target']

x_traing , x_test , y_traing , y_test = train_test_split(x , y ,test_size=0.25,random_state=44,shuffle=True)

print(f"X taring shape is: {x_traing.shape}")
print(f"X test shape is: {x_test.shape}")
print(f"Y taring shape is: {y_traing.shape}")
print(f"Y test shape is: {y_test.shape}")

kerasmodle = keras.models.Sequential([
    keras.layers.Dense(64, activation='tanh'),
    keras.layers.Dense(512,activation='tanh'),
    keras.layers.Dense(32,activation='tanh'),
    keras.layers.Dense(1,activation='sigmoid'),
])

myoptimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.001,
    weight_decay=0.004,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name='Adamw',
   
)

kerasmodle.compile(optimizer=myoptimizer,loss='binary_crossentropy',metrics=['accuracy'])

#Trinag
trinag_modle = kerasmodle.fit(x_traing,y_traing,
                        validation_data=(x_test,y_test),
                        epochs=1000,
                        batch_size=8,
                        verbose=1,
                        callbacks=tf.keras.callbacks.EarlyStopping(
                            patience=20,
                            monitor='val_accuracy',
                            restore_best_weights=True,
                        ))

print(kerasmodle.summary())
kerasmodle.save("heartDisease.keras")
newmodle = keras.models.load_model('heartDisease.keras')
y_pred = newmodle.predict(x_test)

moduleloos,modleaccuracy = newmodle.evaluate(x_test,y_test)

print(f"modle loss is: {moduleloos}")
print(f"modle accuracy is : {modleaccuracy}")

y_pred = [np.round(i[0])for i in y_pred]

          
confusionmatrix = confusion_matrix(y_test, y_pred)
print(f"Confusion_matrix is:\n{confusionmatrix}")          

sns.heatmap(confusionmatrix , center=True , cmap='Blues_r')
#plt.savefig("Confusion.png")

classificationreport = classification_report(y_test , y_pred)
print(f"Classification report is:\n{classificationreport}")


