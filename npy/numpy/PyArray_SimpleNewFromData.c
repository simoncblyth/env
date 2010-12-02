/*
*/

#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>



int main(int argc, char *argv[])
{

     Py_Initialize();
     import_array();

     PyObject *op ;
     op = Py_BuildValue("[(s, s), (s, s)]", "aaaa", "i4", "bbbb", "f4");
     PyObject_Print( op , stdout, 0);
     printf("\n");

     PyArray_Descr* descr;
     PyArray_DescrConverter(op, &descr);
     Py_DECREF(op);

     PyObject *key, *value;
     Py_ssize_t pos = 0;
 
    if(PyDict_Check(descr->fields)){
         printf("print the descr->fields dict ...  %d\n", descr->fields );
         PyObject_Print( (PyObject*)descr->fields , stdout, 0);
         printf("\n\n");

         printf("iterate over the descr->fields dict ...  %d\n", descr->fields );
         while (PyDict_Next( descr->fields , &pos, &key, &value)) {
             PyObject_Print( (PyObject*)key , stdout, 0);
             printf(" ---> ");
             PyObject_Print( (PyObject*)value , stdout, 0);
             printf("\n");

             if(PyTuple_Check(value)){
                   Py_ssize_t s = PyTuple_GET_SIZE(value);
                   if( (int)s >= 2 ){
                       
                       PyObject* ft  = PyTuple_GetItem( value, (Py_ssize_t)0 );
                       printf(" field type ... elsize %d \n", ((PyArray_Descr*)ft)->elsize );
                       PyObject_Print( ft , stdout, 0);
                       printf("\n");

                       PyObject* fo = PyTuple_GetItem( value, (Py_ssize_t)1 );
                       printf(" field offset ... ");
                       PyObject_Print( fo , stdout, 0);
                       printf("\n");
                   }
             }

         }

     } else {
         printf("descr->fields  not dict\n");
     }

    
     int type = PyArray_RegisterDataType( descr );
     printf("type %d\n", type );

     PyArray_Descr* d ;
     d = PyArray_DescrFromType( type );
   /*
    pulling from the register ... yields 
      dtype(('|V8', [('aaaa', '<i4'), ('bbbb', '<f4')]))
    with the dtype wanted embedded in there 


    */

     PyObject_Print( (PyObject*)d , stdout, 0);

     printf("elsize %d\n", d->elsize );
     printf("fields %d\n", d->fields );

     if(PyDict_Check(d->fields)){
         printf("print the fields dict ...  %d\n", d->fields );
         PyObject_Print( (PyObject*)d->fields , stdout, 0);
     } else {
         printf("not dict\n");
     }


     while (PyDict_Next( d->fields , &pos, &key, &value)) {
         PyObject_Print( (PyObject*)key , stdout, 0);
     }




     npy_intp dims[1] ;
     int typenum = type ;
     int nd = 1 ;
     
     dims[0] = 3 ;
     void* data =  NULL ;

     PyObject* a ;
     a = PyArray_SimpleNewFromData( nd,  dims, typenum,  data) ;
     PyObject_Print( (PyObject*)a , stdout, 0);


     return 0;
}



