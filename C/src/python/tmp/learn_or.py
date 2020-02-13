import sys;sys.argv = ['test'] # Здесь обязательно указываем имя скрипта,что бы Matplotlib работал 
import matplotlib.pyplot as plt
import pandas as p

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

# learn or

x=[[1,1],[1,0],[0,1],[0,0] ]
y=[[1],[1],[1],[0]]

map_nn=(2,3,1)

def get_data_x():
  return x

def get_data_y():
  return y

def get_map_nn():
  return map_nn