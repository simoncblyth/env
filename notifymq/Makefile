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


LIBFILE = libnotifymq.$(SOFIX)
OBJ_LIST = example_utils.o notifymq.o
DOBJ_LIST =
NOMQ_LIBS = -L$(LIB_DIR) -lnotifymq

LIBSPATH    = $(RMQC_LIBDIR):$(PRIV_LIBDIR):$(CJSON_LIBDIR):$(LIB_DIR)

all : $(LIB_DIR)/notifymq_sendstring $(LIB_DIR)/notifymq_sendjson

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 


ifdef ROOTSYS
ROOTCFLAGS := $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLIBS   := $(shell $(ROOTSYS)/bin/root-config --libs)
DOBJ_LIST += notifymqDict.o

CFLAGS     += $(ROOTCFLAGS)
LIBS       += $(ROOTLIBS)

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

$(LIB_DIR)/notifymq_sendstring : notifymq_sendstring.c $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(LIBS) $(NOMQ_LIBS) $< -o $@ 

$(LIB_DIR)/notifymq_sendjson : notifymq_sendjson.c $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) $(LIBS) $(NOMQ_LIBS) $(CJSON_LIBS) $< -o $@ 



.PHONY : clean test_sendstring test_sendjson info libspath test_rootsendstring

libspath:
	@echo $(LIBSPATH)

test_rootsendstring:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) root -q -l  tests/test_rootsendstring.C


info:
	@echo $(shell echo $(ENV_HOME))

## run carrot-consumer to provide somewhere for the messages to be consumed 
test_sendstring:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) ./lib/notifymq_sendstring feed importer "$(shell date) $(shell hostname)"

test_sendjson:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) ./lib/notifymq_sendjson runinfo_run runinfo_run_add
		
clean:
	rm -rf $(LIB_DIR) $(DCT_DIR)
