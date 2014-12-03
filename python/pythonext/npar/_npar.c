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

    /*
       TODO
        * hand in the buffer size from an optional argument
        * envvar optional argument with default
        * change name from npar to smth nicer less generic, perhaps qnp
        * arg to control int to float conversion, sometimes want to preserve the
          int bytes rather than loosing info in conversion to float  
        * check if memory really is getting dealloc
    */

    int fbufmax = 1000 ; 
    float* fbuf = (float*)malloc(fbufmax*sizeof(float));

    char type ;
    int nrow ;
    int ncol ; 

    const char* envvar = "SQLITE3_DATABASE" ;
    const char* dbpath = getenv(envvar);
    int rc = querydata(envvar, sql, &nrow, &ncol, &type, fbuf, fbufmax );
    printf("npar: envvar %s:%s ncol %d nrow %d type %c  fbufmax %d  \n", envvar, dbpath, ncol, nrow, type, fbufmax );
    if(rc)
    {
        free(fbuf);
        PyErr_SetString(PyExc_RuntimeError, "querydata error in npar_npar");
        return NULL;
    } 

    /*
    for(int r = 0 ; r < nrow ; r++ )
    {
        for(int c = 0 ; c < ncol ; c++ )
        {
             int index = r*ncol + c ; 
             float f = *(fbuf + index) ; 
             printf("r %d c %d [%f]\n", r, c, f );          
        }
    }
    */

    int typenum ;
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

    PyArrayObject* arr = (PyArrayObject*)PyArray_SimpleNewFromData(nd, dims, typenum, fbuf);
    int flags = PyArray_FLAGS(arr);
    PyArray_ENABLEFLAGS(arr, flags | NPY_ARRAY_OWNDATA ); // get numpy to deallocate when done

    return PyArray_Return(arr);
}


