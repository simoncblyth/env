#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "querydata.h"

static char npar_docstring[] = 
    "Create an numpy ndarray in C";

static char module_docstring[] =
    "This module demonstrates numpy ndarray creation in C";

static PyObject *npar_npar(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"npar", npar_npar, METH_VARARGS, npar_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_npar(void)
{
    PyObject *m = Py_InitModule3("_npar", module_methods, module_docstring);
    if (m == NULL) return;
    import_array();
}


static PyObject* npar_npar(PyObject *self, PyObject *args)
{
    char* sql  = NULL ;
    if(!PyArg_ParseTuple(args, "s", &sql) || sql == NULL) return NULL;


    int typenum ;

    char type ;
    int nrow ;
    int ncol ; 
    void* data = querydata(sql, &nrow, &ncol, &type );

    switch(type)
    {
        case 'f':typenum = NPY_FLOAT32 ;break;
        default: typenum = 0; break ;
    } 

    if(typenum == 0){
        PyErr_SetString(PyExc_RuntimeError, "TypeError in npar_npar");
        return NULL;
    }

    const int nd = 2 ;
    npy_intp dims[nd] = { nrow, ncol };

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, typenum, data);
    int flags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, flags | NPY_ARRAY_OWNDATA ); // get numpy to deallocate when done

    return PyArray_Return(arr);
}


