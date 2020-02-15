/* 
 * File:   PyObjDecl.h
 * Author: papa
 *
 * Created on 13 февраля 2020 г., 9:08
 */

#ifndef PYOBJDECL_H
#define	PYOBJDECL_H
#include "Python.h"
#ifdef	__cplusplus
extern "C" {
#endif
	PyObject *pName = NULL, *pModule = NULL;
	PyObject *pDict = NULL, *pObjct = NULL, *pVal = NULL;
	PyObject* sys = NULL;
	PyObject* sys_path = NULL;
	PyObject* folder_path = NULL;

	PyObject* pDictRandom;
	PyObject* pClassRandom;
	PyObject* pInstanceRandom;
	PyObject* pModuleRandom;



#ifdef	__cplusplus
}
#endif

#endif	/* PYOBJDECL_H */

