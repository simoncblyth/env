
#NAME = PyArray_SimpleNewFromDescr
#NAME = PyArray_FromDims
#NAME = PyArray_RegisterDataType
NAME = PyArray_SimpleNewFromData

PYTHON_LIBDIR := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('LIBDIR') " )
PYTHON_INCDIR := $(shell python -c "from  distutils.sysconfig import get_python_inc ; print get_python_inc() " )
PYTHON_VERSION := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('VERSION') " )

NUMPY_INCDIR := $(shell python -c "import numpy as np ; print np.get_include() ")

MYSQL_INCLUDE := $(shell  mysql_config --include ) 


CFLAGS = -g -I$(PYTHON_INCDIR) -I$(NUMPY_INCDIR) $(MYSQL_INCLUDE)

LIBS = -L$(PYTHON_LIBDIR) -lpython$(PYTHON_VERSION)


%.o : %.c
	@echo "Compiling $< to $@ "
	$(CC) $(CFLAGS) -c $< -o $@ 


$(NAME) : $(NAME).o
	$(CC) $(CFLAGS) $(LIBS)  $< -o $@ 


run :
	./$(NAME)


clean :
	rm *.o $(NAME)

vi :
	vi $(NAME).c

g :
	gdb $(PWD)/$(NAME)


.PHONY : clean run vi g


