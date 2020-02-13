import sys;sys.argv = ['test'] # Здесь обязательно указываем имя скрипта,что бы Matplotlib работал 
"""
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
"""
from struct import pack
push_i=0
push_fl=1
make_kernel=2
stop=b'\x03'
b_c=[]
def py_pack(op_i,val_i_or_fl):
    if op_i==push_fl:
        b_c.append(pack('B',push_fl))
        for i in pack('<f',val_i_or_fl):
            b_c.append(i.to_bytes(1,byteorder='little'))
    elif op_i==push_i:
        b_c.append(pack('B',push_i))
        b_c.append(pack('B',val_i_or_fl))
    elif op_i==make_kernel:
        b_c.append(pack('B',make_kernel))
def dump_bc(f_name):
    b_c.append(stop)
    with open(f_name,'wb') as f:
      for i in b_c:
        f.write(i)

# learn xor
ask_vector=[0,1]
x_max=1

x=[[1,1],[1,0],[0,1],[0,0] ]
y=[[0],[1],[1],[0]]

map_nn=(2,3,1)

def get_data_x():
  return x

def get_data_y():
  return y

def get_map_nn():
  return map_nn

def get_ask_data():
    return ask_vector
    #return [1,1]
def get_x_max_as_koef():
    return x_max
"""
def del_objs():
  del x;del y;del map_nn
"""
  