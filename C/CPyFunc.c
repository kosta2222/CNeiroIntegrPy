#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
//---------------------[Py часть]------------------------
extern PyObject *pName , *pModule;
extern PyObject  *pDict , *pObjct , *pVal ;
extern PyObject* sys ;
extern PyObject* sys_path ;
extern PyObject* folder_path ;

extern PyObject* pDictRandom;
extern PyObject* pClassRandom;
extern PyObject* pInstanceRandom;
extern PyObject* pModuleRandom;

extern int epochs[max_am_epoch];
extern float object_mse[max_am_objMse];
extern int eps;
// Инициализировать интерпретатор Python
void py_init() {
    Py_Initialize();
    Py_DebugFlag = 1;
}

/*
 * Загрузка  модуля (скрипта)
 */
int
python_user_scriptDict(char * py_module_name) {
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
        // Словарь объектов содержащихся в модуле - записывает в глобалную переменную модулей pDict
        pDict = PyModule_GetDict(pModule);
        if (!pDict) return 1;
    $$ while (0);
    // Печать ошибки
    PyErr_Print();
    return 0;
}

/*
 * Освобождение ресурсов интерпритатора python
 */
void
python_clear() {
    // Выгрузка интерпритатора Python
    Py_Finalize();
}

void clear_random() {
    clear_pyObj(pDictRandom);
    clear_pyObj(pClassRandom);
    clear_pyObj(pInstanceRandom);
    clear_pyObj(pModuleRandom);

}

void clear_userModule() {
    clear_pyObj(pDict);
    clear_pyObj(pModule);
    clear_pyObj(pName);
    clear_pyObj(folder_path);
    clear_pyObj(sys_path);
    clear_pyObj(sys);
    clear_pyObj(pVal);
}

void clear_pyObj(PyObject* ob) {
    Py_CLEAR(ob);
}

PyObject*
do_custum_func(PyObject * pDict,const char* func, PyObject * pArgs) {
    PyObject * pVal;
    pObjct = PyDict_GetItemString(pDict, (const char *) func);
    if (!pObjct) return NULL;
    do $
            // Проверка pObjct на годность.
        if (!PyCallable_Check(pObjct)) break;
        pVal = PyObject_CallObject(pObjct, pArgs);
        $$ while (0);
    PyErr_Print();
    return pVal;
}

void plot_grafik_from_C(PyObject* pDict, int * epochs, int eps, float *object_mse) {
    PyObject *py_lst_x, *py_lst_y, *py_tup;
    py_lst_x = PyList_New(eps);
    py_lst_y = PyList_New(eps);
    py_tup = PyTuple_New(2);
    for (int i = 0; i < eps; i++) PyList_SetItem(py_lst_x, i, Py_BuildValue("i", epochs[i]));
    for (int i = 0; i < eps; i++) PyList_SetItem(py_lst_y, i, Py_BuildValue("f", object_mse[i]));
    PyTuple_SetItem(py_tup, 0, py_lst_x);
    PyTuple_SetItem(py_tup, 1, py_lst_y);
    do_custum_func(pDict,"plot_graphic_by_x_and_y", py_tup);
}

void make_matrix_from_pyobj(PyObject *pVal, float* vec, int rows, int cols) {
    PyObject * tmp_row;
    PyObject* tmp_elem;
    float val = 0;
    for (int y = 0; y < rows; y++)$
        tmp_row = PyList_GetItem(pVal, y); // выбираем ряд
        for (int x = 0; x < cols; x++)$
            tmp_elem = PyList_GetItem(tmp_row, x); // выбираем элемент по колонке 		       
        val = (float) PyFloat_AsDouble(tmp_elem);
        vec[y * cols + x] = val;
        $$
        $$
}

void make_vector_from_pyobj(PyObject *pVal, float * vec, int cols) {
    PyObject* tmp_elem;
    float val = 0;
    for (int x = 0; x < cols; x++)$
        tmp_elem = PyList_GetItem(pVal, x); // выбираем элемент из вектора
        incr(tmp_elem);
        val = (float) PyFloat_AsDouble(tmp_elem);
        decr(tmp_elem);
        vec[x] = val;
        $$

}

void initiate_pyRandom_module() {
    pModuleRandom = PyImport_ImportModule("random");
    pDictRandom = PyModule_GetDict(pModuleRandom);
    pClassRandom = PyDict_GetItemString(pDictRandom, "Random");
    pInstanceRandom = PyObject_CallObject(pClassRandom, NULL);
}

PyObject* get_code_objForFunc(PyObject* pDict, char *func) {
    return PyDict_GetItemString(pDict, func);
}

int compil_serializ(PyObject* pDict, nnLay * list, int len_lst, char *f_name) {
    int in = 0;
    int out = 0;
    float matrix[max_in_nn * max_rows_orOut];
    // работаем с функциями скрипта
    pObjct = get_code_objForFunc(pDict, "py_pack");
    for (int l = 0; l < len_lst; l++)$
        in = list[l].in;
        // формируем байт-код 
        PyObject_CallFunction(pObjct, "ii", push_i, in);
        out = list[l].out;
        // формируем байт-код
        PyObject_CallFunction(pObjct, "ii", push_i, out);
        // квадратную матрицу в ленту, потом ее элементы командой в стек
        copy_matrixAsStaticSquare_toRibon(list[l].matrix, matrix, in, out);
        for (int i = 0; i < in * out; i++)
            // формируем байт-код
            PyObject_CallFunction(pObjct, "if", push_fl, matrix[i]);
        // формируем байт-код
        PyObject_CallFunction(pObjct, "ii", make_kernel, 0);
        $$
    pObjct = get_code_objForFunc(pDict, "dump_bc");
    // записываем байты в файл
    PyObject_CallFunction(pObjct, "s", f_name);
    // выводим в консоль ошибки скрипта
    PyErr_Print();
    _0_("compil_serializ");
}

float py_float_to_float(PyObject* pVal) {
    return (float) PyFloat_AsDouble(pVal);
}

int get_list_size(PyObject* listt) {
    return PyList_Size(listt);
}

void create_C_map_nn(PyObject * pVal, int *map_nn, int map_size) {
    PyObject* tmp_elem;
    for (int i = 0; i < map_size; i++)$
        tmp_elem = PyTuple_GetItem(pVal, i);
        map_nn[i] = (int) PyLong_AsLong(tmp_elem);
        decr(tmp_elem);
        $$

}

int get_tuple_sz(PyObject* pVal) {
    return PyTuple_Size(pVal);
}

PyObject* get_list_item(PyObject* pVal, int i) {
    return PyList_GetItem(pVal, i);
}

void incr(PyObject* ob) {

    Py_IncRef(ob);
}

void decr(PyObject* ob) {

    Py_XDECREF(ob);
}
//---------------------[/Py часть]------------------------
