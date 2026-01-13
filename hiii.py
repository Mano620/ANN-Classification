#import → brings a library into your program
# pandas → a library used to work with tables (rows & columns)
# as pd → short name (alias) so we write pd instead of pandas
import pandas as pd 

# sklearn → Scikit-learn (machine learning library)
# model_selection → module for splitting data
# train_test_split → splits data into training and testing parts

from sklearn.model_selection import train_test_split

# preprocessing → used to prepare data
# StandardScaler → scales numbers (mean=0, std=1)
# LabelEncoder → converts text labels into numbers

from sklearn.preprocessing import StandardScaler, LabelEncoder

# pickle → used to save objects to files and load them later

import pickle

# read_csv → reads a CSV file
# Stores the data inside data (a DataFrame)

data = pd.read_csv("Churn_Modelling.csv")

# Shows the first 5 rows
# Used only to check data, not required for training

data.head()

# drop → removes columns
# These columns do not help prediction
# axis=1 → column removal
# (axis=0 would remove rows)

data=data.drop(['RowNumber', 'CustomerId','Surname'],axis=1)

# Creates a LabelEncoder object
# Used to convert text → numbers

label_encoder_gender = LabelEncoder()


data['Gender']= label_encoder_gender.fit_transform(data['Gender'])

from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo = OneHotEncoder()
geo_encoder= onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
onehot_encoder_geo.get_feature_names_out(['Geography'])

geo_encoded_df = pd.DataFrame(
    geo_encoder,
    columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
)

data = pd.concat(
    [data.drop('Geography',axis=1),
     geo_encoded_df],
     axis=1
)
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo, file)

X = data.drop('Exited',axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train =scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler, file)

import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
model =Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1,activation='sigmoid')
])
model.summary()
opt =  tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(
    optimizer=opt,
    loss = "binary_crossentropy",
    metrics= ['accuracy']
)

log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)
early_stopping_callback = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    restore_best_weights = True
)
history = model.fit(
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    epochs=100,
    callbacks=[tensorboard_callback,early_stopping_callback]
)

model.save('mode1.h5')
print("✅ Model training complete and saved successfully!")