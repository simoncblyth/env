
#NAME = PyArray_SimpleNewFromDescr
#NAME = PyArray_FromDims
#NAME = PyArray_RegisterDataType
NAME = PyArray_SimpleNewFromData2
#NAME = PyArray_SimpleNewFromData
#NAME = PyArray_DescrConverter
#NAME = Py_BuildValue
#NAME = dtype
#NAME = datetime
#NAME = datetime_buf
#NAME = types

PYTHON := $(shell which python) 
PYTHON_LIBDIR := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('LIBDIR') " )
PYTHON_LIBPL   := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('LIBPL') " )
PYTHON_VERSION := $(shell python -c "from  distutils.sysconfig import get_config_var ; print get_config_var('VERSION') " )
PYTHON_INCDIR := $(shell python -c "from  distutils.sysconfig import get_python_inc ; print get_python_inc() " )
NUMPY_INCDIR := $(shell python -c "import numpy as np ; print np.get_include() ")
MYSQL_INCLUDE := $(shell  mysql_config --include ) 


CFLAGS = -g -I$(PYTHON_INCDIR) -I$(NUMPY_INCDIR) $(MYSQL_INCLUDE)
LIBS = -L$(PYTHON_LIBPL) -lpython$(PYTHON_VERSION)

%.o : %.c
	@echo "Compiling $< to $@ "
	$(CC) $(CFLAGS) -c $< -o $@ 

$(NAME) : $(NAME).o
	$(CC) $(CFLAGS) $(LIBS)  $< -o $@ 

run :
	./$(NAME)

dyld :
	DYLD_PRINT_LIBRARIES=1 ./$(NAME)


clean :
	rm *.o $(NAME)

vi :
	vi $(NAME).c

g :
	gdb $(PWD)/$(NAME)

info :
	@echo PYTHON $(PYTHON)
	@echo PYTHON_VERSION $(PYTHON_VERSION)
	@echo PYTHON_LIBDIR $(PYTHON_LIBDIR) 
	@echo PYTHON_LIBPL  $(PYTHON_LIBPL) 
	@echo PYTHON_INCDIR $(PYTHON_INCDIR) 
	@echo NUMPY_INCDIR $(NUMPY_INCDIR) 
	@echo MYSQL_INCLUDE $(MYSQL_INCLUDE)




.PHONY : clean run vi g info run dyld


