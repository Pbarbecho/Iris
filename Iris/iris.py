import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# Make results reproducible
seed = 1234
np.random.seed(seed)
#tf.set_random_seed(seed)

# Loading the dataset
dataset = pd.read_csv('/root/NN/Iris/Iris.csv')
dataset = pd.get_dummies(dataset, columns=['Species']) # One Hot Encoding

dataset.to_csv('/root/NN/Iris/encoded_Iris.csv', index=False, header=True)

# Split the dataframe into a training and testing set
X = dataset.iloc[:, 1:-3]
X = scale(X)        # scale values
y = dataset.iloc[:, -3:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# NN model
model=Sequential()
model.add(Dense(1000,input_dim=4,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=10,verbose=1)


# Test model
train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print("Train RMSE: {}".format(train_rmse))
print("Test RMSE: {}".format(test_rmse))
print('------------------------')
scores = model.evaluate(X_train, y_train)
print('Training accuracy: ' , (scores[1]))
scores = model.evaluate(X_test, y_test)
print('Testing accuracy: ' , (scores[1]))

   
y_real = y_test.to_numpy()

col = pd.DataFrame(y_test).keys().to_numpy()
y_predicted_df = pd.DataFrame(test_pred, columns=col)

for c in col:
    y_predicted_df.loc[(y_predicted_df[c] < 0.7), (c)] = 0
    y_predicted_df.loc[(y_predicted_df[c] >= 0.7), (c)] = 1

y_pred = y_predicted_df.to_numpy()

for elem in range(len(test_pred)):
    print(f'Predicted:{y_pred[elem]} Real:{y_real[elem]}')
    
    