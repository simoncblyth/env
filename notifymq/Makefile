include Makefile.$(shell uname)

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



SRC_DIR     = .
INC_DIR     = .
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



LIBFILE = libnotifymq.$(SOFIX)
OBJ_LIST = example_utils.o notifymq.o
DOBJ_LIST =
NOMQ_LIBS = -L$(LIB_DIR) -lnotifymq

LIBSPATH    = $(RMQC_LIBDIR):$(PRIV_LIBDIR):$(CJSON_LIBDIR):$(LIB_DIR)

all : $(LIB_DIR)/mq_sendstring  $(LIB_DIR)/mq_consumebytes $(LIB_DIR)/mq_mapfile $(LIB_DIR)/mq_threaded

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 

$(LIB_DIR)/%.o : $(SRC_DIR)/%.cc
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 



ifdef ROOTSYS
ROOTCFLAGS := $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLIBS   := $(shell $(ROOTSYS)/bin/root-config --libs)
OBJ_LIST  += root2cjson.o MyTMessage.o MQ.o 
DOBJ_LIST += notifymqDict.o root2cjsonDict.o MyTMessageDict.o MQDict.o

CFLAGS     += $(ROOTCFLAGS)
LIBS       += $(ROOTLIBS)
INCLUDES   += $(CJSON_CFLAGS)


$(DCT_DIR)/%Dict.cxx : $(INC_DIR)/%.h $(INC_DIR)/%_$(LINKDEF)
	@echo "Creating $@ from $^ "
	@mkdir -p $(DCT_DIR)
	$(ROOTSYS)/bin/rootcint -f $@ -c $(INCLUDES) $^

$(LIB_DIR)/%.o : $(DCT_DIR)/%.cxx
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ -I.

endif



$(LIB_DIR)/$(LIBFILE) : $(addprefix $(LIB_DIR)/, $(OBJ_LIST)) $(addprefix $(LIB_DIR)/, $(DOBJ_LIST)) 
	@echo "Making $@ from $^ "
	$(CC) $(SOFLAGS) $^ $(LIBS) -o $@

$(LIB_DIR)/mq_sendstring : mq_sendstring.cc $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(LIBS) $(NOMQ_LIBS) $< -o $@ 

$(LIB_DIR)/mq_mapfile : mq_mapfile.cc $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(LIBS) -lNew $(ABDM_CFLAGS) $(NOMQ_LIBS) $(ABDM_LIBS)  $< -o $@ 

$(LIB_DIR)/mq_consumebytes : mq_consumebytes.cc $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) $(ABDM_CFLAGS) $(LIBS) $(NOMQ_LIBS) $(CJSON_LIBS) $(ABDM_LIBS) $< -o $@ 

$(LIB_DIR)/mq_threaded : mq_threaded.cc $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) $(ABDM_CFLAGS) $(LIBS) $(NOMQ_LIBS) $(CJSON_LIBS) $(ABDM_LIBS) $< -o $@ 



.PHONY : clean test_sendstring test_mapfile info libspath test_rootsendstring root ipython test_root2cjson test_root2message test_basic_consume test_consumebytes


root:
	LD_LIBRARY_PATH=$(LIBSPATH) root -l 
ipython:
	LD_LIBRARY_PATH=$(LIBSPATH):$(LD_LIBRARY_PATH) ipython


libspath:
	@echo $(LIBSPATH)


ldd:
	LD_LIBRARY_PATH=$(LIBSPATH):$(LD_LIBRARY_PATH) ldd $(LIB_DIR)/$(LIBFILE)


test_rootsendstring:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) root -q -l  tests/test_rootsendstring.C
test_root2cjson:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) root -q -l  tests/test_root2cjson.C
test_root2message:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) root -q -l  tests/test_root2message.C
test_basic_consume:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) root -q -l  tests/test_basic_consume.C





info:
	@echo $(shell echo $(ENV_HOME))

## run carrot-consumer to provide somewhere for the messages to be consumed 
test_sendstring:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH):$(LD_LIBRARY_PATH) ./lib/mq_sendstring 
test_consumebytes:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH):$(ABDM_LIBDIR):$(LD_LIBRARY_PATH) ./lib/mq_consumebytes
test_mapfile:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH):$(ABDM_LIBDIR):$(LD_LIBRARY_PATH) ./lib/mq_mapfile
test_threaded:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH):$(ABDM_LIBDIR):$(LD_LIBRARY_PATH) ./lib/mq_threaded



		
clean:
	rm -rf $(LIB_DIR) $(DCT_DIR)
