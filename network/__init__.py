import tensorflow as tf
#print(tf.__version__)
from .data_process import train_dataset,validation_dataset,test_dataset,scaled_data_train
from .network_model import model,train_step,validation_step

train_dataset = train_dataset
validation_dataset = validation_dataset
test_dataset = test_dataset

model.predict(scaled_data_train[0:1])

print(train_dataset)
print(model.summary())

#on va définir la fonction de cout et l'optimizer selon notre problème
loss_object = tf.keras.losses.CategoricalCrossentropy()#pas Sparse car one hot encoding
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)#descente de gradient optimisée #learning rate dépendra du nombre de données

#On définit nos metrics pour voir si le réseau se comporte bien
#Pour le perte
train_loss = tf.keras.metrics.Mean(name='train_loss')
validation_loss = tf.keras.metrics.Mean(name='validation_loss')

#Pour la moyenne
#Pas de sparse car one hot encoding
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
validation_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')
