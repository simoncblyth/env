#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


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

    printf("sql %s \n", sql );

    const int nd = 2 ;
    npy_intp dims[nd] = { 2, 2 };

    size_t size = sizeof(float)*dims[0]*dims[1] ;
    float* data = (float*)malloc(size);

    data[0] = 1. ;
    data[1] = 2. ;
    data[2] = 3. ;
    data[3] = 4. ;

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT32, data);
    int flags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, flags | NPY_ARRAY_OWNDATA ); // get numpy to deallocate when done

    return PyArray_Return(arr);
}

/*
PyErr_SetString(PyExc_RuntimeError, "Error in npar_npar");
*/


