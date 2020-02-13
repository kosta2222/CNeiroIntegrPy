#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
#include <stdio.h>
#include <stdlib.h>
#include <synchapi.h>
float koef_to_predict = 0;
int debug = -1;
//========[main функция]=============

int main(int argc, char * argv[]) {
    int eps = 25;
    float X[max_in_nn * max_trainSet_rows];
    float Y[max_rows_orOut * max_trainSet_rows];
    int map_nn[max_am_layer];
    PyObject *inner_list;
    PyObject *pVal;
    int tmp_rows = 0;
    int tmp_cols = 0;
    int cols_teach = 0;
    int cols_train = 0;
    float lr = 0.07;
    int map_size = 0;
    char * main_script = "";
    // получить аргументы из коммандной строки
    if (argc == 5)
        lr = (float) atof(argv[1]), eps = atoi(argv[2]), main_script = argv[3], debug = atoi(argv[4]);
    py_init();
    if (!python_user_script(main_script)) $
        puts("python_init error");
    return -1;
    $$
    //----------Загрузим матрицы из скрипта---------
    /*
     *  Узнаем количество рядов и колонок из скрипта
     */
    /*
     *  Формируем статические вектора обучения и ответов
     */
    /*
     *  Статические вектора идут как С массивы, а получем мы их из Py-обьектов то есть из скрипта
     */
    pVal = do_custum_func("get_data_x", NULL);
    tmp_rows = get_list_size(pVal);
    inner_list = get_list_item(pVal, 0);
    tmp_cols = get_list_size(inner_list);
    cols_train = tmp_cols;
    make_matrix_from_pyobj(pVal, X, tmp_rows, cols_train);
    //	print_deb_matrix(X, tmp_rows, cols_train);
    pVal = do_custum_func("get_data_y", NULL);
    tmp_rows = get_list_size(pVal);
    inner_list = get_list_item(pVal, 0);
    tmp_cols = get_list_size(inner_list);
    cols_teach = tmp_cols;
    make_matrix_from_pyobj(pVal, Y, tmp_rows, cols_teach);
    // используем карту ИНС
    pVal = do_custum_func("get_map_nn", NULL);
    map_size = get_tuple_sz(pVal);
    create_C_map_nn(pVal, map_nn, map_size);
    initiate_pyRandom_module();
    initiate_layers(map_nn, map_size);
    //----------запускаем нейросеть----------
    fit(X, Y, tmp_rows, cols_train, cols_teach, eps, lr, debug);
    //---------------------------------------
    //		plot_grafik_from_C();
    clear_random();
    Sleep(3000);
    printf("Predict:\n");
    pVal = do_custum_func("get_ask_data", NULL);
    tmp_cols = get_list_size(pVal);
    make_vector_from_pyobj(pVal, X, tmp_cols);
    pVal = do_custum_func("get_x_max_as_koef", NULL);
    koef_to_predict = py_float_to_float(pVal);
    predict(X, debug);
    clear_userModule();
    python_clear();
    _0_("main");
}
//========[/main функция]=============

