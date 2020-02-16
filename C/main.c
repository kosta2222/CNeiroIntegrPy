#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
#include "PyObjDecl.h"
#include <stdio.h>
#include <stdlib.h>
//#include <synchapi.h>
//float koef_to_predict = 0;
//int debug = -1;
whole_NN_params NN[1];
int epochs[max_am_epoch];
float object_mse[max_am_objMse];
int eps;
float koef_to_predict;
int debug;
//========[main функция]=============

int main(int argc, char * argv[]) {
    int eps = 25;
    float X[max_in_nn * max_trainSet_rows];
    float Y[max_rows_orOut * max_trainSet_rows];
    float X_test[max_in_nn * max_trainSet_rows];
    float Y_test[max_rows_orOut * max_trainSet_rows];
    int map_nn[max_am_layer];
    PyObject *inner_list;
    int tmp_rows = 0;
    int tmp_cols = 0;
    int cols_teach = 0;
    int cols_train = 0;
    float lr = 0.07;
    int map_size = 0;
    char * script = "";
    char * weight_file = "";
    py_init();
    if (!strcmp("--help", argv[1]))$
        printf("Usage:\nTo train:\t<exe> -train <lr> <epochs> <script to train> <weights file to save> -debug\n");
        printf("To predict:\t<exe> -predict -direct|-bacward <script to ask> <file with weights> -debug\n");
        $$
    // получить аргументы из коммандной строки
    else if (!strcmp(argv[1], "-train") && argc == 7)$
        lr = (float) atof(argv[2]);
        eps = atoi(argv[3]);
        weight_file = argv[5];
        if (!strcmp("-debug", argv[6])) debug = DEBUG;
        script = argv[4];
        if (python_user_scriptDict(script)) $
            puts("get user module error");
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
        pVal = do_custum_func(pDict, "get_data_x", NULL);
        tmp_rows = get_list_size(pVal);
        inner_list = get_list_item(pVal, 0);
        tmp_cols = get_list_size(inner_list);
        cols_train = tmp_cols;
        make_matrix_from_pyobj(pVal, X, tmp_rows, cols_train);
        pVal = do_custum_func(pDict, "get_data_y", NULL);
        tmp_rows = get_list_size(pVal);
        inner_list = get_list_item(pVal, 0);
        tmp_cols = get_list_size(inner_list);
        cols_teach = tmp_cols;
        make_matrix_from_pyobj(pVal, Y, tmp_rows, cols_teach);
        // используем карту ИНС
        pVal = do_custum_func(pDict, "get_map_nn", NULL);
        map_size = get_tuple_sz(pVal);
        create_C_map_nn(pVal, map_nn, map_size);
        initiate_pyRandom_module();
        initiate_layers(map_nn, map_size);
//        //----------настраиваем систему-----------------------------
//        fit(X, Y, tmp_rows, cols_train, cols_teach, eps, lr, debug);
//        //----------------------------------------------------------
        clear_random();
        printf("Cross validation\n");
        pVal = do_custum_func(pDict, "get_data_x_test", NULL);
        tmp_rows = get_list_size(pVal);
        inner_list = get_list_item(pVal, 0);
        tmp_cols = get_list_size(inner_list);
        cols_train = tmp_cols;
        make_matrix_from_pyobj(pVal, X_test, tmp_rows, cols_train);
        pVal = do_custum_func(pDict, "get_data_y_test", NULL);
        tmp_rows = get_list_size(pVal);
        inner_list = get_list_item(pVal, 0);
        tmp_cols = get_list_size(inner_list);
        cols_teach = tmp_cols;
        make_matrix_from_pyobj(pVal, Y_test, tmp_rows, cols_teach);
//        cross_validation(X,Y,tmp_rows,cols_train,cols_teach);
        fit_viaCV(X,Y,X_test,Y_test,80,tmp_rows,cols_train,cols_teach,lr,debug);
        printf("plot\n");
        if (python_user_scriptDict("plot")) $
            puts("get user module error");
            return -1;
            $$
        plot_grafik_from_C(pDict,epochs,eps,object_mse);
        printf("serialize\n");    
        // сохраняем модель
        if (python_user_scriptDict("serial")) $
            puts("get user module error");
            return -1;
            $$
        compil_serializ(pDict, NN->list, NN->nlCount, weight_file);
        clear_userModule();
        python_clear();
        $$
    else if (!strcmp(argv[1], "-predict") && argc == 6)$
        script = argv[3];
        if (python_user_scriptDict(script)) $
            puts("get user module error");
            return -1;
            $$
        if (!strcmp("-debug", argv[5])) debug = DEBUG;
        weight_file = argv[4];
        vm_deserializ(NN->list, weight_file);
        //
        printf("Cross validation\n");
        pVal = do_custum_func(pDict, "get_data_x_test", NULL);
        tmp_rows = get_list_size(pVal);
        inner_list = get_list_item(pVal, 0);
        tmp_cols = get_list_size(inner_list);
        cols_train = tmp_cols;
        make_matrix_from_pyobj(pVal, X, tmp_rows, cols_train);
        pVal = do_custum_func(pDict, "get_data_y_test", NULL);
        tmp_rows = get_list_size(pVal);
        inner_list = get_list_item(pVal, 0);
        tmp_cols = get_list_size(inner_list);
        cols_teach = tmp_cols;
        make_matrix_from_pyobj(pVal, Y, tmp_rows, cols_teach);
        cross_validation(X,Y,tmp_rows,cols_train,cols_teach);
        //
        printf("Predict:\n");
        pVal = do_custum_func(pDict, "get_ask_data", NULL);
        tmp_cols = get_list_size(pVal);
        make_vector_from_pyobj(pVal, X, tmp_cols);
        pVal = do_custum_func(pDict, "get_x_max_as_koef", NULL);
        koef_to_predict = py_float_to_float(pVal);
        printf("koef to pred %f\n", koef_to_predict);
        predict_direct(X, debug);
        clear_userModule();
        python_clear();
        $$


    _0_("main");
}
//========[/main функция]=============

