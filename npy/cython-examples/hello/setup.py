"""

  hello.pyx   looks like python 

def say_hello_to(name):
    print("Hello %s!" % name):

  But gets compiled to C with ...

      python setup.py build_ext --inplace


[blyth@cms01 cython-examples]$ python setup.py build_ext --inplace
running build_ext
cythoning hello.pyx to hello.c
building 'hello' extension
creating build
creating build/temp.linux-i686-2.5
gcc -pthread -fno-strict-aliasing -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/data/env/system/python/Python-2.5.1/include/python2.5 -c hello.c -o build/temp.linux-i686-2.5/hello.o
gcc -pthread -shared build/temp.linux-i686-2.5/hello.o -L/data/env/system/python/Python-2.5.1/lib -lpython2.5 -o hello.so

  Test with :
     from hello import say_hello_to
     say_hello_to("cython")
  

"""

from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Distutils import build_ext 

ext_modules = [
     Extension("hello", 
                ["hello.pyx"],
                libraries=["m"],
             )] 

setup( 
         name = "Hello world app", 
     cmdclass = {'build_ext': build_ext}, 
   ext_modules = ext_modules 
) 

