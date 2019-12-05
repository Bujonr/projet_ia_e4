import tensorflow as tf

class DenseModelV1(tf.keras.Model):#on considère que le réseau n'est pas récurrent

  def __init__(self):
    super(DenseModelV1,self).__init__()
    self.dense1 = tf.keras.layers.Dense(16,name="dense1")
    self.activation = tf.keras.layers.LeakyReLU(alpha=0.3,name="activation")#on utilise leakyrelu pour éviter que des neurones "meurent" car ils ne sont jamais activés
    self.dense2 = tf.keras.layers.Dense(32,name="dense2")
    self.dense3 = tf.keras.layers.Dense(64,name="dense3")
    self.out = tf.keras.layers.Dense(2,activation="sigmoid",name="out")

  def call(self,data):
    dense1 = self.dense1(data)
    activation1 = self.activation(dense1)
    dense2 = self.dense2(activation1)
    activation2 = self.activation(dense2)
    dense3 = self.dense3(activation2)
    out = self.out(dense3)

    return out


model = DenseModelV1()


@tf.function
def train_step(data,label):
  with tf.GradientTape() as tape:
    prediction = model(data)
    loss = loss_object(label,prediction)#on utilise l'objet loss définit plus haut (ici CategoricalCrossentropy)
  gradients = tape.gradient(loss,model.trainable_variables)#calcule le gradient avec la perte et la prediction ci dessus
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))#on utilise l'optimizer définit plus haut (ici Adam)
  train_loss(loss)#On utilise la metric de perte définie plus haut
  train_accuracy(label,prediction)#on utilise la metric de moyenne définie plus haut


#ici la fonction de validation
@tf.function#on convertit automatiquement en mode graphe pour avoir de meilleures performances
def validation_step(data,label):
  prediction = model(data)
  t_loss= loss_object(label,prediction)
  validation_loss(t_loss)
  validation_accuracy(label,prediction)
