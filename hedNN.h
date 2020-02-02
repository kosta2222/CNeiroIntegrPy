/* 
 * File:   hedNN.h
 * Author: papa
 *
 * Created on 9 ноября 2019 г., 23:23
 */
#ifndef HEDNN_H
#define HEDNN_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "hedPy.h"
	// Представляет из себя слой

	typedef struct {
		int in; // сенсоры данного слоя
		int out; // связи-выходы-данного-слоя-синапсы
		float matrix[10][10]; // матрица весов данного слоя
		float cost_signals[10]; // после матричного умножения
		float hidden[10]; // что получилось при функции активации
		float errors[10]; // ошибки данного слоя,их можно сразу наложить на матрицу весов-подправить
	} nnLay;

	typedef struct {
		nnLay list[15];
		int inputNeurons; // количество выходных нейронов
		int outputNeurons; // количество входных нейронов
		int nlCount; // количество слоев
		float inputs[10];
		float targets[10];
		float lr; // коэффициент обучения
	} whole_NN_params;

	typedef enum {
		RELU,
		RELU_DERIV,
		SIGMOID,
		SIGMOID_DERIV,
		TRESHOLD_FUNC,
		TRESHOLD_FUNC_DERIV,
		LEAKY_RELU,
		LEAKY_RELU_DERIV,
		INIT_W_HE,
		INIT_W_GLOROT,
		DEBUG,
		X0
	} OPS;
	//------------------прототипы для обучения-----------------
	float
	sigmoida(float val);
	float
	sigmoidasDerivate(float val);
	void
	backPropagate();
	void
	feedForwarding(bool ok, int debug);
	void
	train(float *in, float *targ, int debug);
	void
	query(float *in);
	int
	getInCount(nnLay *curLay);
	int
	getOutCount(nnLay *curLay);
	float *
	getMatrix(nnLay *curLay);
	void
	updMatrix(nnLay *curLay, float *enteredVal);
	void
	calcHidZeroLay(nnLay* zeroLay, float* targets);
	void
	setIO(nnLay *curLay, int inputs, int outputs);
	void
	init(float lr);
	void
	fit(float *X, float *Y, int rows, int cols_train, int cols_teach, int eps, float lr, int debug);
	void
	makeHidden(nnLay *curLay, float *inputs, int debug);
	float*
	getHidden(nnLay *curLay);
	void
	calcOutError(nnLay *curLay, float *targets);
	void
	calcHidError(nnLay *curLay, float *targets, float *enteredVals);
	float*
	getEssentialGradients(nnLay *curLay);
	float *
	getCostSignals(nnLay *curLay);
	float
	getMinimalSquareError(float *vec, int size_vec);
	float relu(float x);
	float derivateRelu(float x);
	void
	destruct_nn();
	void copy_vector(float *src, float *dest, int n);
	void copy_matrix(float *src, float *dest, int rows, int cols);
	void initiate_layers(int *network_map, int len);
	PyObject* do_custum_func(const char* func, PyObject * pArgs);
	void adaptive_lr(float &mse, float &mse_previous, float &lr, float &lr_previous);
	float py_float_to_float(PyObject* pVal);
	void predict(float* in, int debug);
	float operations(int op, float a, float b, float c, int d, char* str);
	//----------------------------------------------------
#ifdef __cplusplus
}
#endif
#endif /* HEDNN_H */
