/*
api из Python 3.7.3 documentation
*/
glob_vars:PyObject=(pName, pModule\
pDict , pObjct,  \
 sys, \
 sys_path ,\
folder_path )
pVal:PyObject
/*
 инициализируем интерпретатор и
 обеспечиваем чтобы он находил .py скрипт, также
 импортируем модуль и
 узнаем словарь объектов этого модуля(глобальные переменные и функции) и
 возвращаем его
 */
 def  python_init( py_module_name:str)->PyObject
	// Инициализировать интерпретатор Python
	Py_Initialize()
	do :
		/* Загрузка модуля sys
		 Но переменную среды PYTHONHOME не меняем,
		 пусть плагин из этой переменной находит Lib и DLLs, ведь
                 интерпретатор должен функционировать
                 */
		sys = PyImport_ImportModule("sys")
                /* api из Importing Modules */
                /* извлекаем атрибут path(это список) из загруженного модуля*/
		sys_path = PyObject_GetAttrString(sys, "path");
	        /* api из Object Protocol */
                // Создание Py строки
		folder_path = PyUnicode_FromString((const char*) "./src/python");
                /*api из Unicode Objects and Codecs->Creating and accessing Unicode strings*/
                // установке как бы переменной среды PYTHONPATH,добавление
                // строки к списку
		PyList_Append(sys_path, folder_path)
                /*
                 api из List Objects
                */
		// Создание Py строки
		pName = PyUnicode_FromString(py_module_name);
                /* api из  Unicode Objects and Codecs-> Creating and accessing Unicode strings */
		<не получилось>->break
		// Загрузить модуль client
		pModule = PyImport_Import(pName)
                /* api из Importing Modules */
	        <не получилось>->break
		// Словарь объектов содержащихся в модуле этот словарь
                // c так называемым пространством имен модуля
		pDict = PyModule_GetDict(pModule)
                 /* api из Module Objects */
		return pDict
        while (0);
	// Печать ошибки
	PyErr_Print();
vec:vector1D<float>
rows:int
cols:int
def make_matrix_from_pyobj(pVal:PyObject):
 // в pVal список в списке                                                    
 global vec
 vec=malloc(rows*cols)						    
 tmp_row:PyObject
 tmp_elem:PyObject
 val:float
 for y in range(rows):
   tmp_row=PyList_GetItem(pVal,y)// выбираем ряд
   /*
   api из List Objects
   */
   for x in range(cols):
      tmp_elem=PyList_GetItem(tmp_row,x)// выбираем элемент по колонке 
      // извлечь значение
      val=PyFloat_AsDouble(tmp_elem)
      /*
      api из Floating Point Objects
      */
      // вставить в vec
      vec[y*cols+x]=val
def python_clear():
  for i in range(glob_vars):
      // вернуть ресурсы системе                                                 
      Py_XDECREF(i)
      /* api из Reference Counting*/
      // с pVal так не делаем
  // Выгрузка интерпретатора Python
  Py_Finalize()
/*
 выполнить функцию скрипта и получить результат
 */
def do_custum_func( func:str,  pArgs:PyObject)->PyObject:
	pVal:PyObject
	pObjct = PyDict_GetItemString(pDict, func)
        /* api из Dictionary Objects*/
        <не получилось>->return NULL
	do:
			// Проверка pObjct на годность.
			if (!PyCallable_Check(pObjct)):
                        /*api из Object Protocol*/
				break
			/*
                        // Получаем результат кодового обьекта, которого извлекли из pDict                                                                                                 */                          
			pVal = PyObject_CallObject(pObjct, pArgs) 
			/* api из Object Protocol */
        while (0)
	PyErr_Print()
        /* api из Exception Handling */
	return pVal;
}
/*
 Построить график чере .py скрипт plot.py, который принимает 2 списка x и y, у нас
 списки как как C++ vecs
 program graph:
 main->(<обучить модель и получить эпохи и mse>, plot_grafik_from_C)
*/
epochs=vector<int>
mse=vector<float>
def plot_grafik_from_C():
 	( py_lst_x,\
	  py_lst_y,\
	 py_tup):PyObject
        // создание новых списков
         py_lst_x = PyList_New(epochs.size());
	 py_lst_y = PyList_New(mse.size());
          /*
                 api из List Objects
          */
         // создание нового кортежа
         /* в Python скрипте такой кортеж, аргумент
         записывается как *args, то есть кортеж с любым количеством аргументов и
         распаковывается через индексы
         */
	 py_tup = PyTuple_New(2);
         /* api из Tuple Objects */
	 for i in range(len(epochs)):
                // заполнение списков
	       	PyList_SetItem(py_lst_x, i, Py_BuildValue("i", epochs[i]));
                 /*
                 api из List Objects
                */
		}
	 for i in range(len(mse)):
                 // заполнение списков
			PyList_SetItem(py_lst_y, i, Py_BuildValue("f", mse[i]));
                 /*
                 api из List Objects
                */
		}
         // заполнение кортежа
	 PyTuple_SetItem(py_tup, 0, py_lst_x);
	 PyTuple_SetItem(py_tup, 1, py_lst_y);
         /* api из Tuple Objects */
         // вызываем скрипт с аргументами
	 do_custum_func("plot_graphic_by_x_and_y", py_tup);
	 Py_XDECREF(py_lst_x);
	 Py_XDECREF(py_lst_y);
/* Получить размер py списка, для матрицы
последовательно будем передавть внутренние списки через PyList_GetItem(api из List Objects)
*/
def get_list_size(listt:PyObject)->int:
 size:int
 size=PyList_Size(listt)
 /* api из List Objects */
 return size
/* Сделать чтобы слои инициализировались через так называемую карту список*/
def initiate_layers(network_map:list):
 /*ex network_map_c=vector<N>{3,4,2,3,1} */
 len_n_m=sizeof(network_map_c)
 for i in range(len_n_m):
  setIO(network_map_c[i-1],network_map_c[i)
/* у нас есть нн карта map_nn[]={2,3,5,1}
        сделать что бы C хост брал nn_map=(2,3,5,1) из функции Py
        скрипта*/
//py
map_nn=(2,3,5,1)
def get_map_nn():
        return map_nn
//
def main():
        pVal=do_custom_func("get_map_nn",NULL)
        map_size=PyTuple_Size(pVal)
        int *map_nn=new int[map_size]
        tmp_elem:PyObject
        for i in range(map_size):
          tmp_elem=PyTuple_GetItem(pVal,i)
          map_nn[i]=PyLong_AsLong(tmp_elem)
        initiate_layers(map_nn)
/* Сделать адаптивной величину которая называется learning_rate(можно сказать шаг обучения)
        Итак, нам нужна ошибка сети,как известно мерил я ее чере средне-квадратичную ошибку(MSE).По формулам пулучаеся, что нужна delta E,отсюда нам нужна память,чтобы засечь mse(на 4-ой эпохе-текущая) и mse(на 3-ей эпохе)
        Введем память для learning_rate
        Попробуем часть*/

def fit(eps,lr_init):
        NN->lr=lr_init
        mse_t:float
        mse_t_minus_1:float=0
        lr_t:float=NN->lr
        lr_t_minus_1:float=0
        while epocha < nEpochs:
        // перебор пакета
         train(tmp_vec_learn, tmp_vec_targ)
         mse_t=getMinimalSquareError(getHidden(&NN->list[NN->nlCount - 1]), NN->outputNeurons)
         adaptive_lr(mse_t,mse_t_minus_1,lr_t,lr_t_minus_1)
def adaptive_lr(mse_t:<по ссылке>,mse_t_minus_1:<по ссылке>,lr_t:<по ссылке>,lr_t_minus_1:<по ссылке>):
        // для формул
        alpha=0.99
        betta=1.01
        gamma=1.01
         // формула
        delta_E=mse_t-gamma*mse_t_minus_1
        if delta_E>0:
          NN->lr=alpha*lr_t
        else:
          NN->lr=betta*lr_t_minus_1
        mse_t_minus_1=mse_t
        lr_t_minus_1=NN->lr
        
//	if (!PyList_Check(inner_list)) {
//		printf("The argument must be of list or subtype of list");
//
//		return -2;
//	} 
       
program graph:
 main->( python_init<готов словарь модуля>,make_matrix_from_pyobj<использование vec в обучении>)
