/* ИНС с адаптивным коэффициентом обучения и
 * моментумом
 */
#include "hedNN.h"
#include "hedPy.h"
#include <vector>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <signal.h>
using namespace std;
vector<int> epochs;
vector<float>mse;
float *vec; // C вектор из PyObject-а
float *X;
float *Y;
float koef_to_predict;
using namespace std;
//------------------Basic NeiroNet Structures--------------------
whole_NN_params * NN;
//------------------------------------------------------------------
//==================================================================
int main(int argc, char * argv[]) {
	float *X;
	float *Y;
	int tmp_rows;
	int tmp_cols;
	float lr = 0.07;
	int eps = 25;
	char * main_script;
	int debug = -1;
	// получить аргументы из коммандной строки
	if (argc == 5)
		lr = (float) atof(argv[1]), eps = atoi(argv[2]), main_script = argv[3], debug = atoi(argv[4]);
	if (!python_init(main_script))
	{
		puts("python_init error");
		return -1;
	}
	//----------Загрузим матрицы из скрипта---------
	printf("get data x");
	pVal = do_custum_func("get_data_x", NULL);
	tmp_rows = get_list_size(pVal);
	PyObject *inner_list = PyList_GetItem(pVal, 0);
	tmp_cols = get_list_size(inner_list);
	int cols_train = tmp_cols;
	make_matrix_from_pyobj(pVal, tmp_rows, tmp_cols);
	X = vec;
	printf("get data y");
	pVal = do_custum_func("get_data_y", NULL);
	tmp_rows = get_list_size(pVal);
	inner_list = PyList_GetItem(pVal, 0);
	tmp_cols = get_list_size(inner_list);
	int cols_teach = tmp_cols;
	make_matrix_from_pyobj(pVal, tmp_rows, tmp_cols);
	Y = vec;
	// можно пользоваться глобалной vec
	// используем карту НС
	printf("get map nn");
	pVal = do_custum_func("get_map_nn", NULL);
	int map_size = PyTuple_Size(pVal);
	int *map_nn = new int[map_size];
	PyObject* tmp_elem;
	for (int i = 0; i < map_size; i++)
	{
		tmp_elem = PyTuple_GetItem(pVal, i);
		map_nn[i] = (int) PyLong_AsLong(tmp_elem);
		printf("map_nn - %d\n", map_nn[i]);
		decr(tmp_elem);
	}
	initiate_layers(map_nn, map_size);
	initiate_pyRandom_module();
	//----------запускаем нейросеть----------
	fit(X, Y, tmp_rows, cols_train, cols_teach, eps, lr, debug);
	//---------------------------------------
	plot_grafik_from_C();
	printf("Predict:\n");
	pVal = do_custum_func("get_ask_data", NULL);
	tmp_cols = get_list_size(pVal);
	make_vector_from_pyobj(pVal, tmp_cols);
	pVal = do_custum_func("get_x_max_as_koef", NULL);
	koef_to_predict = py_float_to_float(pVal);
	predict(vec, debug);
	python_clear();
	destruct_nn();
	return 0;
}

void initiate_pyRandom_module() {
	pModuleRandom = PyImport_ImportModule("random");
	pDictRandom = PyModule_GetDict(pModuleRandom);
	pClassRandom = PyDict_GetItemString(pDictRandom, "Random");
	pInstanceRandom = PyObject_CallObject(pClassRandom, NULL);
}

float py_float_to_float(PyObject* pVal) {
	return(float) PyFloat_AsDouble(pVal);
}

int get_list_size(PyObject* listt) {
	int size = 0;
	size = PyList_Size(listt);
	return size;
}

void initiate_layers(int *network_map, int size) {
	NN = (whole_NN_params*) new whole_NN_params[1];
	NN->nlCount = size - 1;
	NN->inputNeurons = network_map[0];
	NN->outputNeurons = network_map[NN->nlCount];
	NN->list = (nnLay*) new nnLay[NN->nlCount];
	setIO(&NN->list[0], network_map[1], network_map[0]);
	for (int i = 2; i < size; i++, setIO(&NN->list[i - 1], network_map[i], network_map[i - 1]));
}

void plot_grafik_from_C() {
	PyObject *py_lst_x, *py_lst_y, *py_tup;
	py_lst_x = PyList_New(epochs.size());
	py_lst_y = PyList_New(mse.size());
	py_tup = PyTuple_New(2);
	for (int i = 0; i < epochs.size(); i++, PyList_SetItem(py_lst_x, i, Py_BuildValue("i", epochs[i])));
	for (int i = 0; i < mse.size(); i++, PyList_SetItem(py_lst_y, i, Py_BuildValue("f", mse[i])));
	PyTuple_SetItem(py_tup, 0, py_lst_x);
	PyTuple_SetItem(py_tup, 1, py_lst_y);
	do_custum_func("plot_graphic_by_x_and_y", py_tup);
	decr(py_lst_x);
	decr(py_lst_y);
	decr(py_tup);
}

void print_deb_matrix(float *vec, int rows, int cols) {
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++) printf("%f", vec[i * cols + j]), printf("\n");
	}
}
void make_matrix_from_pyobj(PyObject *pVal, int rows, int cols) {
	PyObject * tmp_row;
	PyObject* tmp_elem;
	float val;
	vec = new float[rows * cols];
	for (int y = 0; y < rows; y++)
	{
		tmp_row = PyList_GetItem(pVal, y); // выбираем ряд
		for (int x = 0; x < cols; x++)
		{
			tmp_elem = PyList_GetItem(tmp_row, x); // выбираем элемент по колонке 		       
			val = (float) PyFloat_AsDouble(tmp_elem);
			vec[y * cols + x] = val;
			decr(tmp_elem);
		}
		decr(tmp_row);
	}
}

void make_vector_from_pyobj(PyObject *pVal, int cols) {
	PyObject* tmp_elem;
	float val;
	vec = new float[cols];
	for (int x = 0; x < cols; x++, tmp_elem = PyList_GetItem(pVal, x), val = (float) PyFloat_AsDouble(tmp_elem), vec[x] = val);
}

//==================================================================
//---------------------Python as Plugin part------------------------
/*
 * Загрузка интерпритатора python и модуля "-//-" в него.
 */
PyObject *
python_init(char * py_module_name) {
	// Инициализировать интерпретатор Python
	Py_Initialize();
	do
	{
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
	} while (0);
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
	// Выгрузка интерпритатора Python
	Py_Finalize();
}

PyObject*
do_custum_func(const char* func, PyObject * pArgs) {
	PyObject * pVal;
	pObjct = PyDict_GetItemString(pDict, (const char *) func);
	if (!pObjct) return NULL;
	do
	{
		{
			// Проверка pObjct на годность.
			if (!PyCallable_Check(pObjct)) break;
			pVal = PyObject_CallObject(pObjct, pArgs);
			if (pVal != NULL) Py_XDECREF(pVal);
			else PyErr_Print();
		}
	} while (0);
	PyErr_Print();
	return pVal;
}

//----------------------------------------------------------------
//---------------------Init,Fit and Destroy NeiroNet--------------
void
destruct_nn() {
	delete(NN);
	delete(NN->list);
}
void
fit(float *X, float *Y, int rows, int cols_train, int cols_teach, int eps, float lr, int debug) {
	NN->lr = lr;
	float mse_t;
	// итерации,обучение
	int nEpoch = eps;
	int epocha = 0;
	// временные вектора для процесса обучения
	float * tmp_vec_x = new float [NN->inputNeurons];
	float * tmp_vec_y = new float [NN->outputNeurons];
	while (epocha < nEpoch)
	{
		//		printf("num Epoch: %d\n", epocha + 1);
		printf("epoch: %d\n", epocha);
		for (int row = 0; row < rows; row++)
		{
			for (int elem = 0; elem < NN->inputNeurons; elem++, tmp_vec_x[elem] = X[row * cols_train + elem]);
			for (int elem = 0; elem < NN->outputNeurons; elem++, tmp_vec_y[elem] = Y[row * cols_teach + elem]);
			train(tmp_vec_x, tmp_vec_y, debug);
			mse_t = getMinimalSquareError(getHidden(&NN->list[NN->nlCount - 1]), NN->outputNeurons);
			printf("mse: %f\n", mse_t);
		}
		mse.push_back(mse_t);
		epocha++;
		epochs.push_back(epocha);
	}
	delete(tmp_vec_x);
	delete(tmp_vec_y);
}

//===============================================================
//--------------------Basic Functions for Learn------------------
void
backPropagate() {
	//-------------------------------ERRORS-----CALC---------
	calcOutError(&NN->list[NN->nlCount - 1], NN->targets);
	calcHidError(&NN->list[NN->nlCount - 1], getEssentialGradients(&NN->list[NN->nlCount - 1]), getCostSignals(&NN->list[NN->nlCount - 1 ]));
	for (int i = NN->nlCount - 2; i > 0; i--, calcHidError(&NN->list[i], getEssentialGradients(&NN->list[i + 1]), getCostSignals(&NN->list[i - 1])));
	// последнему слою не нужны входа т.к. у них нет функции активации
	calcHidZeroLay(&NN->list[0], getEssentialGradients(&NN->list[1]));
	//-------------------------------UPD-----WEIGHT---------
	for (int i = NN->nlCount - 1; i > 0; i--, updMatrix(&NN->list[i], getCostSignals(&NN->list[i - 1])));
	updMatrix(&NN->list[0], NN->inputs);
}

//---------------------------------------------
//---------------------Learn-------------------
/*
обучение с учителем с train set
@param in инфо
@param targ правильный ответ от учител¤
 */
void
train(float *in, float *targ, int debug) {
	NN->inputs = in;
	NN->targets = targ;
	feedForwarding(false, debug);
}

void
predict(float* in, int debug) {
	NN->inputs = in;
	feedForwarding(true, debug);
}

void
feedForwarding(bool ok, int debug) {
	//если ok = true - обучаемся, перед этим выполним один проход по сети
	makeHidden(&NN->list[0], NN->inputs, debug);
	//для данного слоя получить то что отдал пред-слой
	//получаем отдачу слоя и передаем ее следующему  справа как аргумент
	{
		for (int i = 1; i < NN->nlCount; i++, makeHidden(&NN->list[i], getHidden(&NN->list[i - 1]), debug));
	}
	if (ok) for (int out = 0; out < NN->outputNeurons; out++, printf("%d day curs-%f;", out + 1, NN->list[NN->nlCount - 1].hidden[out] * koef_to_predict));
	else backPropagate();
}

int
getInCount(nnLay *curLay) {
	return curLay->in;
}

int
getOutCount(nnLay *curLay) {
	return curLay->out;
}

float **
getMatrix(nnLay *curLay) {
	return curLay->matrix;
}

void
updMatrix(nnLay *curLay, float *enteredVal) {
	// 0.1 регуляризатор;+ 0.8 * curLay->matrix[row][elem]-как я думаю моментум
	for (int row = 0; row < curLay->out; row++)
		for (int elem = 0; elem < curLay->in; elem++, curLay->matrix[row][elem] -= NN->lr * curLay->errors[elem] * enteredVal[elem]);
}

#define randWeight(sum_of_neurons,a,b) ( ((float)rand() / (float)RAND_MAX)* (b-a)+a);
#define randWeight1(in) ( ((float)rand() / (float)RAND_MAX)* sqrt(2.0/in));
#define randWeight2(in,a,b) (((float)rand() / (float)RAND_MAX)* (b*sqrt(2.0/in)-a*sqrt(2.0/in))+a*sqrt(2.0/in));
//* pow(out,-0.5)))

void
setIO(nnLay *curLay, int outputs, int inputs) {
	{
		// сенсоры
		curLay->in = inputs + 1;
		// нейроны-выходы-синапсы
		curLay->out = outputs;
		// отдача нейронов
		curLay->hidden = new float [curLay->out];
		curLay->cost_signals = new float [curLay->out];
		curLay->matrix = (float**) malloc((curLay->out) * sizeof(float));
	}
	for (int row = 0; row < curLay->out; row++)
	{
		curLay->matrix[row] = (float*) malloc(curLay->in * sizeof(float));
	}
	for (int row = 0; row < curLay->out; row++)
	{
		for (int elem = 0; elem < curLay->in; elem++)
		{
			curLay->matrix[row][elem] = operations(INIT_W_HE, 0, 0, 0, "");
		}
	}
}

void
makeHidden(nnLay *curLay, float *inputs, int debug) {
	float tmpS = 0.0;
	float val = 0;
	for (int row = 0; row < curLay->out; row++)
	{
		for (int elem = 0; elem < curLay->in; elem++)
			if (elem == 0) tmpS += curLay->matrix[row][0];
			else tmpS += curLay->matrix[row][elem] * inputs[elem];
		curLay->cost_signals[row] = tmpS, val = relu(tmpS), curLay->hidden[row] = val, operations(debug, curLay->cost_signals[row], 0, 0, "cost signals");
		tmpS = 0;
	}
	//	}
	operations(debug, 0, 0, 0, "make hidden made");
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
	{
		curLay->errors = (float*) malloc((curLay->out) * sizeof(float));
	}
	for (int row = 0; row < curLay->out; row++, curLay->errors[row] = (curLay->hidden[row] - targets[row]) * derivateRelu(curLay->cost_signals[row]));
}

void
calcHidError(nnLay *curLay, float *essential_gradients, float *enteredVals) {
	{
		curLay->errors = (float*) malloc((curLay->in) * sizeof(float));
	}
	for (int elem = 0; elem < curLay->in; elem++)
	{
		{
			curLay->errors[elem] = 0.0;
		}
		for (int row = 0; row < curLay->out; row++, curLay->errors[elem] += essential_gradients[row] * curLay->matrix[row][elem] * derivateRelu(enteredVals[elem]));
	}
}

void
calcHidZeroLay(nnLay* zeroLay, float * essential_gradients) {
	{
		zeroLay->errors = (float*) malloc((zeroLay->in) * sizeof(float));
	}
	for (int elem = 0; elem < zeroLay->in; elem++)
	{
		{
			zeroLay->errors[elem] = 0.0;
		}
		for (int row = 0; row < zeroLay->out; row++, zeroLay->errors[elem] += essential_gradients[row] * zeroLay->matrix[row][elem]);
	}
}

float*
getEssentialGradients(nnLay *curLay) {
	return curLay->errors;
}

float
getMinimalSquareError(float *vec, int size_vec) {
	float sum;
	for (int row = 0, sum = 0; row < size_vec; row++, sum += vec[row]);
	float square = pow(sum, 2);
	float mean = square / size_vec;
	return mean;
}

float
sigmoida(float val) {
	float res = (1.0 / (1.0 + exp(val)));
	//	if(isnan(res)) return 0;
	return res;
}

float
sigmoidasDerivate(float val) {
	float res = exp(-val) / (pow((1 + exp(-val)), 2));
	//	if(isnan(res)) return 0;
	return res;
}

float relu(float x) {
	if (x < 0)
		//		return 0.001 * abs(x);
		return 0;
	else
		//		return 0.001*x;
		return x;
}

float derivateRelu(float x) {
	if (x < 0)
		//		return -0.001;
		return 0;
	else
		//		return 1*0.001;
		return 1;
}

void incr(PyObject* ob) {
	Py_IncRef(ob);
}

void decr(PyObject* ob) {
	Py_XDECREF(ob);
}

float operations(int op, float a, float b, float c, char* str) {
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
		//print(strr,":",a,"\n")
		printf("%s : %f\n", str, a);
		break;
	}
	case INIT_W_HE:
	{
		PyObject* pValue;
		float r;
		//		return((float) rand() / (float) RAND_MAX)*(b * sqrt(2 / c) - a * sqrt(2 / c)) + a * sqrt(2 / c);
		if (PyCallable_Check(pInstanceRandom)) pValue = PyObject_CallMethod(pInstanceRandom, "gauss", "ff", 0.0, 1.0);
		else
			PyErr_Print();
		if (pValue != NULL) r = PyFloat_AsDouble(pValue), decr(pValue), printf("r he:%f", r);
		else
			PyErr_Print();
		return r * sqrt(2 / a);
	}
	}
}
//--------------------------------------------------------------
