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

#define max_in_nn 30
#define max_trainSet_rows 200
#define max_validSet_rows 50	
#define max_rows_orOut	10
#define max_am_layer 7
#define max_am_epoch 25
#define max_am_objMse max_am_epoch

	typedef unsigned char u_char;
#define max_stack_matrEl 256
#define max_stack_otherOp 4
#define bin_kernel_bufLen 256*3	

	// Представляет из себя слой

	typedef struct {
		int in; // сенсоры данного слоя
		int out; // связи-выходы-данного-слоя-синапсы
		float matrix[max_rows_orOut][max_in_nn]; // матрица весов данного слоя
		float cost_signals[max_rows_orOut]; // после матричного умножения
		float hidden[max_rows_orOut]; // что получилось при функции активации
		float errors[max_rows_orOut]; // ошибки данного слоя,их можно сразу наложить на матрицу весов-подправить
	} nnLay;

	typedef struct {
		nnLay list[max_am_layer];
		int inputNeurons; // количество выходных нейронов
		int outputNeurons; // количество входных нейронов
		int nlCount; // количество слоев
		float inputs[max_in_nn];
		float targets[max_rows_orOut];
		float lr; // коэффициент обучения
	} whole_NN_params;

	// Байт-код для обучения сети

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
		DEBUG_STR
	} OPS;

	// байт-коды для сериализации/десериализации-загрузка входов/выходов,загрузка элементов матрицы,сворачивание то есть создания ядра,остановка ВМ

	typedef enum {
		push_i,
		push_fl,
		make_kernel,
		stop
	};
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
	float relu(float x);
	float derivateRelu(float x);
	float
	getMinimalSquareError(float *out_nn, float* teacher_answ, int size_vec);
	void copy_vector(float *src, float *dest, int n);
	void copy_matrixAsRibons(float *src, float *dest, int rows, int cols);
	void print_deb_matrixAsRibon(float *vec, int rows, int cols, char * label);
	void print_deb_vector(float *vec, int cols, char * label);
	int copy_matrixAsStaticSquare_toRibon(float src[][max_in_nn], float *dest, int in, int out);
	void initiate_layers(int *network_map, int len);
	void predict_direct(float* in, int debug);
	float operations(int op, float a, float b, float c, int d, char* str);
	//----------------------------------------------------
#ifdef __cplusplus
}
#endif
#endif /* HEDNN_H */
