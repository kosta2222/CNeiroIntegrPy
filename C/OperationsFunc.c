#include "hedNN.h"
#include "hedPy.h"
#include "utilMacr.h"
//-----------------[Операция наподобии виртуальной машины]------------
extern PyObject * pInstanceRandom;
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
            PyObject * pVal;
            float r = 0;
            pVal = PyObject_CallMethod(pInstanceRandom, "gauss", "ii", 0, 1);
            if (pVal != NULL) r = PyFloat_AsDouble(pVal), clear_pyObj(pVal), printf("r he:%f\n", r);
            else PyErr_Print();
            decr(pVal);
            return r * sqrt(2 / a);
        }
        case DEBUG_STR:
        {
           printf("%s\n",str);

        }
    }
}
//-----------------[Операция наподобии виртуальной машины]------------