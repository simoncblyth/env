

#include <Python.h>
#include <stdio.h>


int main(int argc, char *argv[])
{
     Py_Initialize();
     PyObject* o ;

     //o = Py_BuildValue("{s:i,s:i}", "abc", 123, "def", 456);
     //o = Py_BuildValue("{s:s,s:s}", "abc", "123", "def", "456");
     //o = Py_BuildValue("{s:s,s:s}", "abc", "123", "def", "456");
     //o = Py_BuildValue("{s:(si),s:(si)}", "abc", "123",10, "def", "456",20 );
     o = Py_BuildValue("s", "123" );

     PyObject_Print( o , stdout, 0 );
     printf("\n");
     return 0 ;
}  

