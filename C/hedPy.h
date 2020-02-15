/* 
 * File:   hedPy.h
 * Author: papa
 *
 * Created on 10 ноября 2019 г., 20:08
 */
#ifndef HEDPY_H
#define	HEDPY_H
#include "Python.h"

void py_init();
int
python_user_scriptDict(char * py_module_name);
void
python_clear();
void
python_func_get_str(char *val);
int
python_func_get_val(char *val);
PyObject*
do_custum_func(PyObject* pDict, const char* func, PyObject* pArgs);
int get_list_size(PyObject* listt);
void make_vector_from_pyobj(PyObject *pVal, float *vec, int cols);
void make_matrix_from_pyobj(PyObject *pVal, float *, int, int);
void plot_grafik_from_C(PyObject* pDict, int * epochs, int eps, float *object_mse);
void print_deb_matrix(float *vec, int rows, int cols);
void initiate_pyRandom_module();
int get_tuple_sz(PyObject* pVal);
PyObject* get_list_item(PyObject* pVal, int i);
void incr(PyObject* ob);
void decr(PyObject* ob);
void clear_random();
void clear_userModule();
void clear_pyObj(PyObject* ob);
void create_C_map_nn(PyObject * pVal, int *map_nn, int map_size);
float py_float_to_float(PyObject* pVal);
int compil_serializ(PyObject* pDict, nnLay * list, int len_lst, char *f_name);
PyObject* get_code_objForFunc(PyObject* pDict, char *func);
#endif	/* HEDPY_H */
