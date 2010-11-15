
       4 levels ..

           0)   mysql.pxd : C API/library supplied by MYSQL
                   wrapper to allow cython to understand the base MYSQL types

           1)  _mysql.pxd : _mysql CPython module (from MySQL-python adustman)
                   wrapper for access to _mysql low level MySQL-python functionality   (not the higher level MySQLdb yet ... for sanity)
                   connect/query/get_result

           2)  npmysql.pyx  Cython extension module

                   Cython extension module
                      ... aiming to add numpy array result fetching to this 
                     

           3) test_npmysql.py    python usage of the extension module





