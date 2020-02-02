/* 
 * File:   hedPy.h
 * Author: papa
 *
 * Created on 10 ноября 2019 г., 20:08
 */
#ifndef HEDPY_H
#define	HEDPY_H
extern "C" {
#include <Python.h>
}
PyObject *pName = NULL, *pModule = NULL;
PyObject *pDict = NULL, *pObjct = NULL, *pVal = NULL;
PyObject* sys = NULL;
PyObject* sys_path = NULL;
PyObject* folder_path = NULL;

PyObject* pDictRandom;
PyObject* pClassRandom;
PyObject* pInstanceRandom;
PyObject* pModuleRandom;
void py_init();
PyObject *
python_user_script(char *);
void
python_clear();
void
python_func_get_str(char *val);
int
python_func_get_val(char *val);
PyObject*
do_custum_func(const char* func, PyObject* pArgs);
int get_list_size(PyObject* listt);
void make_vector_from_pyobj(PyObject *pVal, int);
void make_matrix_from_pyobj(PyObject *pVal, float *, int, int);
void plot_grafik_from_C();
void print_deb_matrix(float *vec, int rows, int cols);
void initiate_pyRandom_module();
int get_tuple_sz(PyObject* pVal);
PyObject* get_list_item(PyObject* pVal, int i);
void incr(PyObject* ob);
void decr(PyObject* ob);
void create_C_map_nn(PyObject * pVal, int *map_nn, int map_size);
#endif	/* HEDPY_H */
