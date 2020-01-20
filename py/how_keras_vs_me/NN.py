import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
import matplotlib.pyplot as plt

#map_nn=(28,12,13,14,7)

def plot_history(history):
    fig,ax=plt.subplots()
    x=range(20)
    plt.plot(x,history.history['loss'])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mse")
    plt.show()

np.random.seed(5)

# learn xor.Потом использую 
#x=np.array([[1,1],[1,0],[0,1],[0,0] ])
#y=np.array([[0],[1],[1],[0]])

# learn many
x=np.random.normal(size=(170,28))
y=np.random.normal(size=(170,7))


def get_data_x():
  return x

def get_data_y():
  return y

#map_nn=(28,12,13,14,7) аналогия с мое map nn

model=Sequential()
model.add(Dense(12,input_dim=28,activation='sigmoid'))
model.add(Dense(13,activation='sigmoid'))
model.add(Dense(14, activation='sigmoid'))
model.add(Dense(7, activation='sigmoid'))

model.compile(optimizer='SGD',loss='mse',metrics=['mse'])
history=model.fit(get_data_x(),get_data_y(),epochs=20)
plot_history(history)


