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
INCLUDES    = -I$(INC_DIR)

CFLAGS     += $(RMQC_CFLAGS)
LIBS       += $(RMQC_LIBS)

CFLAGS     += $(PRIV_CFLAGS)
LIBS       += $(PRIV_LIBS)


LIBFILE = libnotifymq.$(SOFIX)
OBJ_LIST = example_utils.o notifymq.o
NOMQ_LIBS = -L$(LIB_DIR) -lnotifymq

LIBSPATH    = $(RMQC_LIBDIR):$(PRIV_LIBDIR):$(CJSON_LIBDIR):$(LIB_DIR)

all : $(LIB_DIR)/notifymq_sendstring $(LIB_DIR)/notifymq_sendjson

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 

$(LIB_DIR)/$(LIBFILE) : $(addprefix $(LIB_DIR)/, $(OBJ_LIST)) 
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



.PHONY : clean test_sendstring test_sendjson info

info:
	@echo $(shell echo $(ENV_HOME))

## run carrot-consumer to provide somewhere for the messages to be consumed 
test_sendstring:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) ./lib/notifymq_sendstring feed importer "$(shell date) $(shell hostname)"

test_sendjson:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) ./lib/notifymq_sendjson feed importer "$(shell date) $(shell hostname)"
		
clean:
	rm -rf $(LIB_DIR)
