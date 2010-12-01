
NAME = PyArray_SimpleNewFromDescr

PYTHON_LIBDIR := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('LIBDIR') " )
PYTHON_INCDIR := $(shell python -c "from  distutils.sysconfig import get_python_inc ; print get_python_inc() " )
PYTHON_VERSION := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('VERSION') " )

NUMPY_INCDIR := $(shell python -c "import numpy as np ; print np.get_include() ")


CFLAGS = -I$(PYTHON_INCDIR) -I$(NUMPY_INCDIR)

LIBS = -L$(PYTHON_LIBDIR) -lpython$(PYTHON_VERSION)


%.o : %.c
	@echo "Compiling $< to $@ "
	$(CC) $(CFLAGS) -c $< -o $@ 


$(NAME) : $(NAME).o
	$(CC) $(CFLAGS) $(LIBS)  $< -o $@ 


clean :
	rm *.o $(NAME)


.PHONY : clean


