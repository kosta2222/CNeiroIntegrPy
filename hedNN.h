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
        float** matrix; // матрица весов данного слоя
        float *cost_signals; // после матричного умножения
        float* hidden; // что получилось при функции активации
        float* errors; // ошибки данного слоя,их можно сразу наложить на матрицу весо-подправить
    } nnLay;
    //------------------прототипы для обучения-----------------
    float
    sigmoida(float val);
    float
    sigmoidasDerivate(float val);
    void
    backPropagate();
    void
    feedForwarding(bool ok);
    void
    train(float *in, float *targ);
    void
    query(float *in);
    int
    getInCount(nnLay *curLay);
    int
    getOutCount(nnLay *curLay);
    float **
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
    fit(int epochs, float lr);
    void
    makeHidden(nnLay *curLay, float *inputs);
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
    destruct();
    void make_matrix_from_pyobj(PyObject* pVal);
    void initiate_layers(int *network_map, int len);
    PyObject* do_custum_func(const char* func, PyObject * pArgs);
    void adaptive_lr(float &mse, float &mse_previous, float &lr, float &lr_previous);
    void make_vector_from_pyobj(PyObject *pVal);
    float py_float_to_float(PyObject* pVal);
    void predict(float* in);
    float operations(int op, float a, float b, float c, char* str);
    //----------------------------------------------------
#ifdef __cplusplus
}
#endif
#endif /* HEDNN_H */
