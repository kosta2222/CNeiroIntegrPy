/* 
 * File:   main.cpp
 * Author: papa
 *
 * Created on 2 ноября 2019 г., 11:29
 */
#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int debug = 10;
#define DEBUG 10
#define c(f,id) {debug==DEBUG?printf("%s\n",id),f:f;} 
#define for_in(cn,from,to)\
;int cn=0;\
for(cn=from;cn<to;cn++)
#define $ {
#define $$ }
#define lbr printf("[")
#define rbr printf("]\n")
#define nl printf("\n")
#define _0_(func) \
lbr;\
printf("Exit succes\n");\
printf("in func->%s\n",func);\
rbr;\
return 0
#define _e_(func,var) \
printf("in func->%s\n",func);printf("var->%s\n",var);
#define ms0 printf ("Static memory error\n")
#define md0 printf ("Dinamic memory error\n")
#define ne0 printf ("Null error\n")
#define check_d(r,func,var) if(r==0) {lbr;md0;_e_(func,var);rbr;}/*if макрос работает только на одной строке*/
#define check_s(i,max_buf,func,var) if(i>max_buf){lbr;ms0;_e_(func,var);rbr;}
#define is_null_ptrErr(op,func,var) if(op==0) {lbr;ne0;_e_(func,var);rbr;}
void copy_vector(float *src, float *dest, int n);
#define max_in_nn 30
#define max_trainSet_rows 200
#define max_rows_orOut	10
#define max_am_layer 7
#define max_am_epoch 25
#define max_am_objMse max_am_epoch	
// Представляет из себя слой
typedef struct {
    int in; // сенсоры данного слоя
    int out; // связи-выходы-данного-слоя-синапсы
    float matrix[max_rows_orOut][max_in_nn]; // матрица весов данного слоя
    float cost_signals[max_rows_orOut]; // после матричного умножения
    float hidden[max_rows_orOut]; // что получилось при функции активации
    float errors[max_rows_orOut]; // ошибки данного слоя,их можно сразу наложить на матрицу весов-подправить
} nnLay;
typedef unsigned char u_char;
#define max_stack_matrEl 256
#define max_stack_otherOp 4
#define bin_kernel_bufLen 256*3
void py_init();
int compil_serializ(nnLay * list, int len_lst, char * f_name);
PyObject*
do_custum_func(const char* func, PyObject * pArgs);
// получить исполняемый кодовый обьект(функцию) из словаря модуля
PyObject* get_code_objForFunc(PyObject* pDict, char *func);
int make_kernel_f(nnLay *list, int lay_pos, float *matrix_el_st, int * ops_st, int sp_op);
int vm(nnLay *list, int len, u_char *bin_buf);
int vm_deserializ(nnLay * list, int len, char* f_name);
// копирование квадратной матрицы в ленту
int copy_matrix_as_vec(float src[][max_in_nn], float *dest, int in, int out);
PyObject *
python_user_script(char * py_module_name);
/* Копирование статических массиввов например для локального контекста*/
void copy_vector(float *src, float *dest, int n);
PyObject *pObjct_fu1, *pDict, *pArgs, *sys, *sys_path, *folder_path, *pName, *pModule;
// байт-коды-загрузка входов/выходов,загрузка элементов матрицы,сворачивание то есть создания ядра,остановка ВМ
typedef enum {
    push_i,
    push_fl,
    make_kernel,
    stop
};
// лента ядер для инициализации и сериализации
nnLay list[max_am_layer];
// лента ядер после десериализации
nnLay list_des[max_am_layer]; 

int main() {
    // тест инициализация ядер
    for_in(i, 0, 3)$
    list[i].in = 2;
    list[i].out = 3;
    for (int row = 0; row < list[i].out; row++)
        for (int elem = 0; elem < list[i].in; elem++)
            list[i].matrix[row][elem] = 5.0;
    $$
    py_init();
    // сериализация
    compil_serializ(list, 3, "./test1.bin");
    // десериализация и создание ленты ядер, матриц
    vm_deserializ(list_des, 3, "./test1.bin");
    // тест десериализации
    for_in(j, 0, 3)$
    for (int row = 0; row < list_des[j].out; row++)
        for (int elem = 0; elem < list_des[j].in; elem++)
            printf("deser matrix el:%f\n", list_des[j].matrix[row][elem]);
    $$
    _0_("main");
}

void py_init() {
    Py_Initialize();
    Py_DebugFlag = 1;
}

int compil_serializ(nnLay * list, int len_lst, char *f_name) {
    int in = 0;
    int out = 0;
    float matrix[max_in_nn * max_rows_orOut];
    // работаем со скриптом
    PyObject * pDict = python_user_script("py_pack_mod");
    // работаем с функциями скрипта
    pObjct_fu1 = get_code_objForFunc(pDict, "py_pack");
    for (int l = 0; l < len_lst; l++)$
        in = list[l].in;
    // формируем байт-код 
    PyObject_CallFunction(pObjct_fu1, "ii", push_i, in);
    out = list[l].out;
    // формируем байт-код
    PyObject_CallFunction(pObjct_fu1, "ii", push_i, out);
    // квадратную матрицу в ленту, потом ее элементы командой в стек
    copy_matrix_as_vec(list[l].matrix, matrix, in, out);
    for (int i = 0; i < in * out; i++)
        // формируем байт-код
        PyObject_CallFunction(pObjct_fu1, "if", push_fl, matrix[i]);
    // формируем байт-код
    PyObject_CallFunction(pObjct_fu1, "ii", make_kernel, 0);
    $$
    pObjct_fu1 = get_code_objForFunc(pDict, "dump_bc");
    // записываем байты в файл
    PyObject_CallFunction(pObjct_fu1, "s", f_name);
    // выводим в консоль ошибки скрипта
    PyErr_Print();
    _0_("compil_serializ");
}

void copy_vector(float *src, float *dest, int n) {
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

int copy_matrix_as_vec(float src[][max_in_nn], float *dest, int in, int out) {
    for (int row = 0; row < out; row++)
        for (int elem = 0; elem < in; elem++)
            dest[row * in + elem] = src[row][elem];
    _0_("copy_matrix_as_vec");
}

int make_kernel_f(nnLay *list, int lay_pos, float *matrix_el_st, int * ops_st, int sp_op) {
    int out = ops_st[sp_op];
    int in = ops_st[sp_op - 1];
    list[lay_pos].out = out;
    list[lay_pos].in = in;
    for (int row = 0; row < out; row++)
        for (int elem = 0; elem < in; elem++)
            list[lay_pos].matrix[row][elem] = matrix_el_st[row * elem];
    _0_(" make_kernel");
}

int vm(nnLay *list, int len, u_char *bin_buf) {
    float matrix_el_st[max_stack_matrEl];
    int ops_st[max_stack_otherOp];
    int ip = 0;
    int sp_ma = -1;
    int sp_op = -1;
    u_char op = -1;
    float arg = 0;
    int n_lay = 0;
    op = bin_buf[ip];
    while (op != stop) {
        switch (op) {
                // загружаем на стек количество входов и выходов ядра
            case push_i:
            {
                ops_st[++sp_op] = bin_buf[++ip];
                nl;
                break;
            }
                // загружаем на стек элементы матриц
            case push_fl:
            {
                arg = *((float *) (&bin_buf[ip + 1]));
                matrix_el_st[++sp_ma] = arg;
                ip += 4;
                break;
            }
                // создаем одно ядро в массиве
            case make_kernel:
            {
                printf("print steck in vm in push_fl\n");
                for (int i = 0; i < 6; i++) printf("matrix el:%f", matrix_el_st[i]);
                make_kernel_f(list, n_lay, matrix_el_st, ops_st, sp_op);
                // переходим к следующему индексу ядра
                n_lay++;
                // зачищаем стеки
                sp_op = -1;
                sp_ma = -1;
                break;
            }
        }
        // показываем на следующую инструкцию
        ip++;
        op = bin_buf[ip];
    }
    _0_("vm");
}

int vm_deserializ(nnLay * list, int len, char* f_name) {
    u_char bin_buf[bin_kernel_bufLen];
    FILE * fp;
    fp = fopen(f_name, "rb");
    is_null_ptrErr(fp,"vm_deserializ","fp");
    check_s(ftell(fp), bin_kernel_bufLen, "vm_deserializ", "bin_kernel_bufLen");
    while (fread(bin_buf, 1, 256, fp) != NULL);
    fclose(fp);
    // разборка байт-кода
    vm(list, len, bin_buf);
    _0_("vm_deserializ");
}

/*
 * Загрузка  модуля (скрипта)
 */
PyObject *
python_user_script(char * py_module_name) {
    do $
        // Загрузка модуля sys
        // Но переменную среды PYTHONHOME не меняем,
        // пусть плагин из этой переменной находит Lib и DLLs
        sys = PyImport_ImportModule("sys");
    sys_path = PyObject_GetAttrString(sys, "path");
    // Путь до наших исходников python
    // То,что строит график лежит в <где exe-шник>/src/python/plot.py
    folder_path = PyUnicode_FromString("./src/python");
    PyList_Append(sys_path, folder_path);
    // Создание Unicode объекта из UTF-8 строки
    pName = PyUnicode_FromString(py_module_name);
    if (!pName) break;
    // Загрузить модуль client
    pModule = PyImport_Import(pName);
    if (!pModule) break;
    // Словарь объектов содержащихся в модуле
    pDict = PyModule_GetDict(pModule);
    if (!pDict) break;
    return pDict;
    $$ while (0);
    // Печать ошибки
    PyErr_Print();
}

PyObject*
do_custum_func(const char* func, PyObject * pArgs) {
    PyObject * pVal;
    pObjct_fu1 = PyDict_GetItemString(pDict, (const char *) func);
    if (!pObjct_fu1) return NULL;
    do $
            // Проверка pObjct на годность.
        if (!PyCallable_Check(pObjct_fu1)) break;
    pVal = PyObject_CallObject(pObjct_fu1, pArgs);
    $$ while (0);
    PyErr_Print();
    return pVal;
}

PyObject* get_code_objForFunc(PyObject* pDict, char *func) {
    return PyDict_GetItemString(pDict, func);
}
