#include "PyObjDecl.h"
#include "hedNN.h"
//#include "hedPy.h"
#include "utilMacr.h"
//---------------------[Py часть]------------------------
// Инициализировать интерпретатор Python
void py_init() {
    Py_Initialize();
    Py_DebugFlag = 1;
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
do_custum_func(const char* func, PyObject * pArgs) {
    PyObject * pVal;
    pObjct = PyDict_GetItemString(pDict, (const char *) func);
    if (!pObjct) return NULL;
    do $
            // Проверка pObjct на годность.
        if (!PyCallable_Check(pObjct)) break;
    pVal = PyObject_CallObject(pObjct, pArgs);
    /*
        if (pVal != NULL) Py_XDECREF(pVal);
        else PyErr_Print();
     */
    $$ while (0);
    PyErr_Print();
    return pVal;
}

void plot_grafik_from_C() {
    PyObject *py_lst_x, *py_lst_y, *py_tup;
    py_lst_x = PyList_New(eps);
    py_lst_y = PyList_New(eps);
    py_tup = PyTuple_New(2);
    for (int i = 0; i < eps; i++) PyList_SetItem(py_lst_x, i, Py_BuildValue("i", epochs[i]));
    for (int i = 0; i < eps; i++) PyList_SetItem(py_lst_y, i, Py_BuildValue("f", object_mse[i]));
    PyTuple_SetItem(py_tup, 0, py_lst_x);
    PyTuple_SetItem(py_tup, 1, py_lst_y);
    do_custum_func("plot_graphic_by_x_and_y", py_tup);
}

void make_matrix_from_pyobj(PyObject *pVal, float* vec, int rows, int cols) {
    PyObject * tmp_row;
    PyObject* tmp_elem;
    float val = 0;
    for (int y = 0; y < rows; y++) $
        tmp_row = PyList_GetItem(pVal, y); // выбираем ряд
    for (int x = 0; x < cols; x++) $
        tmp_elem = PyList_GetItem(tmp_row, x); // выбираем элемент по колонке 		       
    val = (float) PyFloat_AsDouble(tmp_elem);
    vec[y * cols + x] = val;
    $$
    $$
}

void make_vector_from_pyobj(PyObject *pVal, float * vec, int cols) {
    PyObject* tmp_elem;
    float val = 0;
    for (int x = 0; x < cols; x++) $
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

float py_float_to_float(PyObject* pVal) {
    return (float) PyFloat_AsDouble(pVal);
}

int get_list_size(PyObject* listt) {
    return PyList_Size(listt);
}

void create_C_map_nn(PyObject * pVal, int *map_nn, int map_size) {
    PyObject* tmp_elem;
    for (int i = 0; i < map_size; i++) {
        tmp_elem = PyTuple_GetItem(pVal, i);
        map_nn[i] = (int) PyLong_AsLong(tmp_elem);
        decr(tmp_elem);
    }

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
