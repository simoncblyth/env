
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
AALIBSPATH   = $(ALIBSPATH):$(ARCH_LIBDIR)
XLIBSPATH   = 



all : $(LIB_DIR)/mq_sendstring  $(LIB_DIR)/mq_consumebytes $(LIB_DIR)/mq_mapfile $(LIB_DIR)/mq_threaded $(LIB_DIR)/mq_monitor

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
	cd $(LIB_DIR) && ln -s lib$(LIBNAME).dylib lib$(LIBNAME).so
endif

$(LIB_DIR)/mq_sendstring : mq_sendstring.cc $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CCX) $(CFLAGS) $(INCLUDES) $(LIBS) $(NOMQ_LIBS) $< -o $@ 



.PHONY : clean test_sendstring test_mapfile info libspath test_rootsendstring root ipython test_root2cjson test_root2message test_basic_consume test_consumebytes test_load otool


root:
	LD_LIBRARY_PATH=$(LIBSPATH) root -l 
ipython:
	LD_LIBRARY_PATH=$(LIBSPATH):$(LD_LIBRARY_PATH) ipython


libspath:
	@echo $(LIBSPATH)


ldd:
	LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ldd $(LIB_DIR)/lib$(LIBNAME).so

otool:
	DYLD_LIBRARY_PATH=$(ALIBSPATH):$(DYLD_LIBRARY_PATH) otool -L $(LIB_DIR)/lib$(LIBNAME).dylib
       

test_rootsendstring:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_rootsendstring.C
test_root2cjson:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_root2cjson.C
test_root2message:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_root2message.C
test_basic_consume:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_basic_consume.C
test_load:
	DYLD_LIBRARY_PATH=$(ALIBSPATH) LD_LIBRARY_PATH=$(ALIBSPATH) root -q -l  tests/test_load.C


test_evmq:
	DYLD_LIBRARY_PATH=$(ALIBSPATH):$(DYLD_LIBRARY_PATH) LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ipython evmq.py
imq:
	DYLD_LIBRARY_PATH=$(ALIBSPATH):$(DYLD_LIBRARY_PATH) LD_LIBRARY_PATH=$(ALIBSPATH):$(LD_LIBRARY_PATH) ipython tests/mq.py

info:
	@echo $(shell echo $(ENV_HOME))

## run carrot-consumer to provide somewhere for the messages to be consumed 
test_sendstring:
	DYLD_LIBRARY_PATH=$(LIBSPATH):$(DYLD_LIBRARY_PATH) LD_LIBRARY_PATH=$(LIBSPATH):$(LD_LIBRARY_PATH) ./lib/mq_sendstring 


		
clean:
	rm -rf $(LIB_DIR) $(DCT_DIR)
