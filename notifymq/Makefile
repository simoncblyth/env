
ARCH := $(shell uname)
include Makefile.$(ARCH)

RMQC_HOME   = $(LOCAL_BASE)/env/messaging/rabbitmq-c
RMQC_LIBDIR = $(RMQC_HOME)/librabbitmq/.libs
RMQC_CFLAGS = -I$(RMQC_HOME)/librabbitmq  
RMQC_LIBS   = -L$(RMQC_LIBDIR) -lrabbitmq

PRIV_HOME   = $(ENV_HOME)/priv
PRIV_LIBDIR = $(PRIV_HOME)/lib
PRIV_CFLAGS = -I$(PRIV_HOME)
PRIV_LIBS   = -L$(PRIV_LIBDIR) -lprivate

CJSON_HOME   = $(LOCAL_BASE)/env/messaging/cjson
CJSON_LIBDIR = $(CJSON_HOME)/lib
CJSON_CFLAGS = -I$(CJSON_HOME)  
CJSON_LIBS   = -L$(CJSON_LIBDIR) -lcJSON

ABDM_HOME    = $(ABERDEEN_HOME)/DataModel
ABDM_LIBDIR  = $(ABDM_HOME)/lib
ABDM_CFLAGS  = -I$(ABDM_HOME)/include
ABDM_LIBS    = -L$(ABDM_LIBDIR) -lAbtDataModel

# tis failing ... so get from Makefile.arch
#GLIB_CFLAGS   = $(shell pkg-config --cflags gthead-2.0)
#GLIB_LIBS     = $(shell pkg-config --libs gthread-2.0

NOMQ_LIBDIR = $(PWD)/lib



SRC_DIR     = src
INC_DIR     = include
LIB_DIR     = lib
DCT_DIR     = dict
LINKDEF     = LinkDef.hh
INCLUDES    = -I$(INC_DIR)

CFLAGS     += $(RMQC_CFLAGS)
LIBS       += $(RMQC_LIBS)

CFLAGS     += $(PRIV_CFLAGS)
LIBS       += $(PRIV_LIBS)

CFLAGS     += $(CJSON_CFLAGS)
LIBS       += $(CJSON_LIBS)

CFLAGS     += $(GLIB_CFLAGS)
LIBS       += $(GLIB_LIBS)




LIBNAME = notifymq
LIBFILE = lib$(LIBNAME).$(SOFIX)
OBJ_LIST = example_utils.o notifymq.o notifymq_utils.o notifymq_collection.o
DOBJ_LIST =
NOMQ_LIBS = -L$(LIB_DIR) -lnotifymq

LIBSPATH    = $(RMQC_LIBDIR):$(PRIV_LIBDIR):$(CJSON_LIBDIR):$(NOMQ_LIBDIR)
ALIBSPATH   = $(LIBSPATH):$(ABDM_LIBDIR)



all : $(LIB_DIR)/mq_sendstring  

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ 

$(LIB_DIR)/%.o : $(SRC_DIR)/%.cc
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ 



ifdef ROOTSYS
ROOTCFLAGS := $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLIBS   := $(shell $(ROOTSYS)/bin/root-config --libs)
OBJ_LIST  += root2cjson.o MyTMessage.o MQ.o 
DOBJ_LIST += notifymqDict.o root2cjsonDict.o MyTMessageDict.o MQDict.o

CFLAGS     += $(ROOTCFLAGS)
LIBS       += $(ROOTLIBS)
INCLUDES   += $(CJSON_CFLAGS)
INCLUDES   += $(RMQC_CFLAGS)
endif

$(DCT_DIR)/%Dict.cxx : $(INC_DIR)/%.h $(INC_DIR)/%_$(LINKDEF)
	@echo "Creating $@ from $^ "
	@mkdir -p $(DCT_DIR)
	$(ROOTSYS)/bin/rootcint -f $@ -c $(INCLUDES) $^

$(LIB_DIR)/%.o : $(DCT_DIR)/%.cxx
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ -I.


$(LIB_DIR)/lib$(LIBNAME).$(SOFIX) : $(addprefix $(LIB_DIR)/, $(OBJ_LIST)) $(addprefix $(LIB_DIR)/, $(DOBJ_LIST)) 
	@echo "Making $@ from $^ "
	$(CC) $(SOFLAGS) $^ $(LIBS) -o $@   
ifeq ($(ARCH), Darwin)
	@echo "Create .so symbolic link to .dylib for Darwin workaround "
	cd $(LIB_DIR) && ln -sf lib$(LIBNAME).dylib lib$(LIBNAME).so
endif

$(LIB_DIR)/mq_sendstring : mq_sendstring.cc $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CCX) $(CFLAGS) $(INCLUDES) $(LIBS) $(NOMQ_LIBS) $< -o $@ 



.PHONY : info libspath root ipython ldd otool test_load test_sendstring test_sendjson test_sendobj evmq imq test_sendstring_ clean 

info:
	@echo $(shell echo $(ENV_HOME))
libspath:
	@echo $(LIBSPATH)
root:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -l 
ipython:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ipython
ldd:
	LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ldd $(LIB_DIR)/lib$(LIBNAME).so
otool:
	DYLD_LIBRARY_PATH=$(ALIBSPATH):$(DYLD_LIBRARY_PATH) otool -L $(LIB_DIR)/lib$(LIBNAME).dylib


test_load:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_load.C       
test_sendstring:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_sendstring.C
test_sendjson:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_sendjson.C
test_sendobj:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_sendobj.C


evmq:
	DYLD_LIBRARY_PATH=$(ALIBSPATH):$(DYLD_LIBRARY_PATH) LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ipython evmq.py
imq:
	DYLD_LIBRARY_PATH=$(ALIBSPATH):$(DYLD_LIBRARY_PATH) LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ipython tests/mq.py


test_sendstring_:
	DYLD_LIBRARY_PATH=$(LIBSPATH):$(DYLD_LIBRARY_PATH) LD_LIBRARY_PATH=$(LIBSPATH):$(LD_LIBRARY_PATH) ./lib/mq_sendstring 
		
clean:
	rm -rf $(LIB_DIR) $(DCT_DIR)
