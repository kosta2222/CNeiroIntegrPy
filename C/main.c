/* ИНС с адаптивным коэффициентом обучения и
 * моментумом
 */
#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
//#include <vector>
#include <stdio.h>
#include <stdlib.h>
//using namespace std;
//vector<int> epochs;
//vector<float>mse;
float koef_to_predict = 0;
//------------------Основная структура ИНС--------------------
whole_NN_params NN[1];
//------------------------------------------------------------
int epochs[max_am_epoch];
float object_mse[max_am_objMse];
int eps = 25;
//========[main функция]=============

int main(int argc, char * argv[]) {
    float X[max_in_nn * max_trainSet_rows];
    float Y[max_rows_orOut * max_trainSet_rows];
    int map_nn[max_am_layer];
    PyObject *inner_list;
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
    //	print_deb_matrix(Y, tmp_rows, cols_teach);
    // используем карту ИНС
    pVal = do_custum_func("get_map_nn", NULL);
    map_size = get_tuple_sz(pVal);
    create_C_map_nn(pVal, map_nn, map_size);
    //    initiate_pyRandom_module();
    c(initiate_pyRandom_module(), "initiate_pyRandom_module()")
    //    initiate_layers(map_nn, map_size);
    c(initiate_layers(map_nn, map_size), "initiate_layers(map_nn, map_size)")
    //	print_deb_matrix(Y, tmp_rows, cols_teach);
    //----------запускаем нейросеть----------
    c(fit(X, Y, tmp_rows, cols_train, cols_teach, eps, lr, debug), "fit(X, Y, tmp_rows, cols_train, cols_teach, eps, lr, debug)")
    //---------------------------------------
    //		plot_grafik_from_C();
    printf("Predict:\n");
    pVal = do_custum_func("get_ask_data", NULL);
    tmp_cols = get_list_size(pVal);
    make_vector_from_pyobj(pVal, X, tmp_cols);
    //	pVal = do_custum_func("get_x_max_as_koef", NULL);
    //	koef_to_predict = py_float_to_float(pVal);
    predict(X, debug);
    /*
        python_clear();
     */
    _0_("main");
}
//========[/main функция]=============

void copy_vector(float *src, float *dest, int n) {
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

void copy_matrix(float *src, float *dest, int rows, int cols) {
    for (int row = 0; row < rows; row++)
        for (int elem = 0; elem < cols; elem++) dest[row * cols + elem] = src[row * cols + elem];
}

void initiate_layers(int *network_map, int size) {
    int in = 0;
    int out = 0;
    NN->nlCount = size - 1;
    NN->inputNeurons = network_map[0];
    NN->outputNeurons = network_map[NN->nlCount];
    //    setIO(&NN->list[0], network_map[0], network_map[1]);
    c(setIO(&NN->list[0], network_map[0], network_map[1]), "setIO(&NN->list[0], network_map[0], network_map[1])")
    for (int i = 2; i <= NN->nlCount; i++)
        in = network_map[i - 1], out = network_map[i], setIO(&NN->list[i - 1], in, out), printf("in: %d \t out:%d\n", in, out);

}

void print_deb_matrix(float *vec, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) printf("%f", vec[i * cols + j]), printf("\n");
}

//---------------------[Py часть]------------------------
// Инициализировать интерпретатор Python

void py_init() {
    Py_Initialize();
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
    // Вернуть ресурсы системе
    decr(pDict);
    decr(pModule);
    decr(pName);
    decr(folder_path);
    decr(sys_path);
    decr(sys);
    decr(pDictRandom);
    decr(pClassRandom);
    decr(pInstanceRandom);
    decr(pModuleRandom);
    // Выгрузка интерпритатора Python
    Py_Finalize();
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
    if (pVal != NULL) Py_XDECREF(pVal);
    else PyErr_Print();
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
    val = (float) PyFloat_AsDouble(tmp_elem);
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

//---------------------[Fit Инс]--------------------------

void
fit(float *X, float *Y, int rows, int cols_train, int cols_teach, int eps, float lr, int debug) {
    NN->lr = lr;
    float mse_after_oneVector = 0;
    int epocha = 0;
    // итерации,обучение
    // временные вектора для процесса обучения
    float tmp_vec_x[NN->inputNeurons];
    float tmp_vec_y[NN->outputNeurons];
    while (epocha < eps)$
        printf("epoch: %d\n", epocha);
    for (int row = 0; row < rows; row++)$
        for (int elem = 0; elem < NN->inputNeurons; elem++) tmp_vec_x[elem] = X[row * cols_train + elem];
    for (int elem = 0; elem < NN->outputNeurons; elem++) tmp_vec_y[elem] = Y[row * cols_teach + elem];
    //    train(tmp_vec_x, tmp_vec_y, debug);
    c(train(tmp_vec_x, tmp_vec_y, debug), "train(tmp_vec_x, tmp_vec_y, debug)")
    mse_after_oneVector = getMinimalSquareError(getHidden(&NN->list[NN->nlCount - 1]), Y, NN->outputNeurons);
    printf("mse: %f\n", mse_after_oneVector);
    if (mse_after_oneVector == 0) goto out_bach;
    $$
    /*
     *  Все векторы из пакета отдали, запишем последнюю ошибку
     */
    object_mse[epocha] = mse_after_oneVector;
    epochs[epocha] = epocha;
    epocha++;
    $$
    out_bach :;

}
//---------------------[Fit Инс]--------------------------

//-----------------[Основные функции обучения]------------

void
backPropagate() {

    /* Вычисление ошибки */c(calcOutError(&NN->list[NN->nlCount - 1], NN->targets), "calcOutError(&NN->list[NN->nlCount - 1], NN->targets)");

    c(calcHidError(&NN->list[NN->nlCount - 1], getEssentialGradients(&NN->list[NN->nlCount - 1]), getCostSignals(&NN->list[NN->nlCount - 1 ])), "calcHidError(&NN->list[NN->nlCount - 1], getEssentialGradients(&NN->list[NN->nlCount - 1]), getCostSignals(&NN->list[NN->nlCount - 1 ]))");

    for (int i = NN->nlCount - 2; i > 0; i--) c(calcHidError(&NN->list[i], getEssentialGradients(&NN->list[i + 1]), getCostSignals(&NN->list[i - 1])), "calcHidError(&NN->list[i], getEssentialGradients(&NN->list[i + 1]), getCostSignals(&NN->list[i - 1]))");
    /* Последнему слою не нужны входа т.к. у них нет функции активации */
    c(calcHidZeroLay(&NN->list[0], getEssentialGradients(&NN->list[1])), "calcHidZeroLay(&NN->list[0], getEssentialGradients(&NN->list[1]))");
    /* Обновление весов */for (int i = NN->nlCount - 1; i > 0; i--)c(updMatrix(&NN->list[i], getCostSignals(&NN->list[i - 1])), "updMatrix(&NN->list[i], getCostSignals(&NN->list[i - 1]))");
    c(updMatrix(&NN->list[0], NN->inputs), "updMatrix(&NN->list[0], NN->inputs)");
}

void
train(float *in, float *targ, int debug) {
    /*
     *  Работает с рядом из матриц X и Y
     */
    copy_vector(in, NN->inputs, max_in_nn);
    copy_vector(targ, NN->targets, max_rows_orOut);
    c(feedForwarding(false, debug), "feedForwarding(false, debug)")


}

void
predict(float* in, int debug) {
    /*
     *  Работает с одним вектром
     */
    copy_vector(in, NN->inputs, max_in_nn);
    feedForwarding(true, debug);
}

void
feedForwarding(bool ok, int debug) {
    // если ok = true - обучаемся, перед этим выполним один проход по сети
    c(makeHidden(&NN->list[0], NN->inputs, debug), "makeHidden(&NN->list[0], NN->inputs, debug)")
            // для данного слоя получить то что отдал пред-слой
            // получаем отдачу слоя и передаем ее следующему  справа как аргумент
    for (int i = 1; i < NN->nlCount; i++) makeHidden(&NN->list[i], getHidden(&NN->list[i - 1]), debug);
    if (ok) for (int out = 0; out < NN->outputNeurons; out++) printf("%d item val %f;", out + 1, NN->list[NN->nlCount - 1].hidden[out] /** koef_to_predict*/);
    else c(backPropagate(), "backPropagate()")
    }

int
getInCount(nnLay *curLay) {

    return curLay->in;
}

int
getOutCount(nnLay *curLay) {

    return curLay->out;
}

//float *
//getMatrix(nnLay *curLay) {
//        float matrix[(curLay->out)*(curLay->in)];
//	copy_matrix(matrix,curLay->matrix,curLay->out,curLay->in);
//	return matrix;
//}

void
updMatrix(nnLay *curLay, float *enteredVal) {
    for (int row = 0; row < curLay->out; row++)
        for (int elem = 0; elem < curLay->in; elem++, curLay->matrix[row][elem] -= NN->lr * curLay->errors[elem] * enteredVal[elem]);
}

void
setIO(nnLay *curLay, int inputs, int outputs) {
    /* сенсоры - входа*/curLay->in = inputs + 1;
    /* данный ряд нейронов */curLay->out = outputs;
    for (int row = 0; row < curLay->out; row++)for (int elem = 0; elem < curLay->in; elem++)printf("operations\n"), curLay->matrix[row][elem] = operations(INIT_W_HE, curLay->in, 0, 0, 0, "");

}

void
makeHidden(nnLay *curLay, float *inputs, int debug) {
    float tmpS = 0;
    float val = 0;
    for (int row = 0; row < curLay->out; row++) {
        for (int elem = 0; elem < curLay->in; elem++)
            if (elem == 0) tmpS += curLay->matrix[row][0];
            else tmpS += curLay->matrix[row][elem] * inputs[elem];
        curLay->cost_signals[row] = tmpS, val = relu(tmpS), curLay->hidden[row] = val, operations(debug, curLay->cost_signals[row], 0, 0, 0, "cost signals");
        tmpS = 0;
    }
    operations(debug, 0, 0, 0, 0, "make hidden made");
}

float *
getCostSignals(nnLay * curLay) {

    return curLay->cost_signals;
}

float*
getHidden(nnLay *curLay) {

    return curLay->hidden;
}

void
calcOutError(nnLay *curLay, float *targets) {
    for (int row = 0; row < curLay->out; row++, curLay->errors[row] = (curLay->hidden[row] - targets[row]) * derivateRelu(curLay->cost_signals[row]));
}

void
calcHidError(nnLay *curLay, float *essential_gradients, float *enteredVals) {
    for (int elem = 0; elem < curLay->in; elem++) for (int row = 0; row < curLay->out; row++, curLay->errors[elem] += essential_gradients[row] * curLay->matrix[row][elem] * derivateRelu(enteredVals[elem]));
}

void
calcHidZeroLay(nnLay* zeroLay, float * essential_gradients) {
    for (int elem = 0; elem < zeroLay->in; elem++) for (int row = 0; row < zeroLay->out; row++) zeroLay->errors[elem] += essential_gradients[row] * zeroLay->matrix[row][elem];
}

float*
getEssentialGradients(nnLay *curLay) {
    return curLay->errors;
}

float
getMinimalSquareError(float *out_nn, float* teacher_answ, int size_vec) {
    float sum = 0;
    float square = 0;
    float mean = 0;
    for (int row = 0, sum = 0; row < size_vec; row++) sum += out_nn[row] - teacher_answ[row];
    square = pow(sum, 2);
    mean = square / size_vec;
    return mean;
}

float
sigmoida(float val) {
    float res = (1.0 / (1.0 + exp(val)));
    return res;
}

float
sigmoidasDerivate(float val) {
    float res = exp(-val) / (pow((1 + exp(-val)), 2));
    return res;
}

float relu(float x) {
    if (x < 0)
        return 0;
    else
        return x;
}

float derivateRelu(float x) {
    if (x < 0)
        return 0;
    else
        return 1;
}
//-----------------[/Основные функции обучения]--------------

//-----------------[Операция наподобии виртуальной машины]---

float operations(int op, float a, float b, float c, int d, char* str) {
    switch (op) {
        case RELU:
        {
            if (a < 0)
                return 0;
            else
                return a;
        }
        case RELU_DERIV:
        {
            if (a < 0)
                return 0;
            else
                return a;
        }
        case TRESHOLD_FUNC:
        {
            if (a < 0)
                return 0;
            else
                return 1;
        }
        case TRESHOLD_FUNC_DERIV:
        {
            return 0;
        }
        case LEAKY_RELU:
        {
            if (a < 0)
                return b * a;
            else
                return a;
        }
        case LEAKY_RELU_DERIV:
        {
            if (a < 0)
                return b;
            else
                return 1;
        }
        case SIGMOID:
        {
            return 1.0 / (1 + exp(b * (-a)));
        }
        case SIGMOID_DERIV:
        {
            return b * 1.0 / (1 + exp(b * (-a)))*(1 - 1.0 / (1 + exp(b * (-a))));
        }
        case DEBUG:
        {
            printf("%s : %f\n", str, a);
            break;
        }
        case INIT_W_HE:
        {
            float r;
            pVal = PyObject_CallMethod(pInstanceRandom, "gauss", "ii", 0, 1);
            check_d(pVal, "operations INIT_W_HE", "pVal");
            if (pVal != NULL) r = PyFloat_AsDouble(pVal), /*decr(pValue)*/ printf("r he:%f\n", r);
            else PyErr_Print();
            return r * sqrt(2 / a);
        }
        case X0:
        {
            if (d == 0) printf("Null pointer exception-%s\n", str);

        }
    }
}
//-----------------[Операция наподобии виртуальной машины]---