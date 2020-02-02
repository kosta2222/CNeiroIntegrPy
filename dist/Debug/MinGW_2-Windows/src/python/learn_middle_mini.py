import sys;sys.argv = ['test'] # Здесь обязательно указываем имя скрипта,что бы Matplotlib работал 
import matplotlib.pyplot as plt 
import numpy as np


def plot_graphic_by_x_and_y(*args):
  fig,ax=plt.subplots() 
  ax.plot(args[0],args[1])
  ax.grid()
  ax.set_xlabel("epocha",
          fontsize=15,
          color='red',
          bbox={'boxstyle':'rarrow',
              'pad':0.1,
              'edgecolor':'red',
              'linewidth':3})
  ax.set_ylabel("mse",
          fontsize=15,
          color='red',
          bbox={'boxstyle':'rarrow',
              'pad':0.1,
              'edgecolor':'red',
              'linewidth':3})
  fig.set_figwidth(9)
  fig.set_figheight(9)
  plt.show()

map_nn=(28,10,7)

np.random.seed(5)

# learn many
x=np.random.normal(size=(40,28)).tolist()
y=np.random.normal(size=(40,7)).tolist()

def get_data_x():
  return x

def get_data_y():
  return y

def get_map_nn():
  return map_nn