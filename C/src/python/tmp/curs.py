import sys;sys.argv = ['test'] # Здесь обязательно указываем имя скрипта,что бы Matplotlib работал
import matplotlib.pyplot as plt
import csv

import random
import pandas as pd
import math
import numpy as np
# Читаем временной ряд курса валюты
df=pd.read_excel(".\\RC_2.xlsx")
# Обрабатываем временной ряд через точки с периодами в прошлое-28 дней и будущее-7 дней
money=df["curs"]
past=7*4
future=7

end=len(money)-future

stat=past
raw_df=[]

for i in range(stat,end):
    past_and_futures_values=money[i-past:i+future]
    raw_df.append(list(past_and_futures_values))

past_columns=[]
for i in range(past):
    past_columns.append("past_{}".format(i))

future_columns = []
for i in range(future):
    future_columns.append("future_{}".format(i))
# формируем данные для отдачи НС
df_nn = pd.DataFrame(raw_df, columns=(past_columns + future_columns))
x=df_nn[past_columns][:-20];y=df_nn[future_columns][:-20]
x = x.to_numpy()
y = y.to_numpy()
x_max = x.max()
y_max = y.max()
x_norm=x/x_max
y_norm=y/y_max
x_test=df_nn[past_columns][-20:]
y_test=df_nn[future_columns][-20:]
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()
x_test_max = x_test.max()
y_test_max = y_test.max()
x_test_norm=x_test/x_test_max
y_tes_norm=y_test/y_test_max

x_norm=x_norm.tolist()
y_norm=y_norm.tolist()

ask_vector=[]
# В формате csv Exel данные через запятую
with open("ask.csv","r") as f:
  # Поменяем разделитель чтобы во float ах осталась запятая,как было в данных Exel
  reader_csv=csv.reader(f,delimiter=';')
  for i in reader_csv:
     curs_val= i[0].replace(',','.')
     ask_vector.append(float(curs_val))



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


# Карта слоев
map_nn=(28,9,8,7)

def get_data_x():
    print("test ",type(x_norm)," len(x_norm)",len(x_norm),"len(x_norm[0]",len(x_norm[0]))
    return x_norm

def get_data_y():
    return y_norm



def get_map_nn():
    return map_nn

def get_x_max_as_koef():
    return x_max
def get_ask_data():
  return ask_vector