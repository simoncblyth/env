include Makefile.$(shell uname)

RMQC_HOME   = $(LOCAL_BASE)/env/messaging/rabbitmq-c
RMQC_LIBDIR = $(RMQC_HOME)/librabbitmq/.libs
RMQC_CFLAGS = -I$(RMQC_HOME)/librabbitmq  
RMQC_LIBS   = -L$(RMQC_LIBDIR) -lrabbitmq

PRIV_HOME   = $(ENV_HOME)/priv
PRIV_LIBDIR = $(PRIV_HOME)/lib
PRIV_CFLAGS = -I$(PRIV_HOME)
PRIV_LIBS   = -L$(PRIV_LIBDIR) -lprivate


SRC_DIR     = .
INC_DIR     = .
LIB_DIR     = lib
INCLUDES    = -I$(INC_DIR)

CFLAGS     += $(RMQC_CFLAGS)
LIBS       += $(RMQC_LIBS)

CFLAGS     += $(PRIV_CFLAGS)
LIBS       += $(PRIV_LIBS)


LIBFILE = libnotifymq.$(SOFIX)
OBJ_LIST = example_utils.o
NOMQ_LIBS = -L$(LIB_DIR) -lnotifymq

LIBSPATH    = $(RMQC_LIBDIR):$(PRIV_LIBDIR):$(LIB_DIR)

all : $(LIB_DIR)/amqp_sendstring

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 

$(LIB_DIR)/$(LIBFILE) : $(addprefix $(LIB_DIR)/, $(OBJ_LIST)) 
	@echo "Making $@ from $^ "
	$(CC) $(SOFLAGS) $^ $(LIBS) -o $@

$(LIB_DIR)/amqp_sendstring : amqp_sendstring.c $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(LIBS) $(NOMQ_LIBS) $< -o $@ 

.PHONY : clean test info

info:
	@echo $(shell echo $(ENV_HOME))

test:
	DYLD_LIBRARY_PATH=$(LIBSPATH) LD_LIBRARY_PATH=$(LIBSPATH) ./lib/amqp_sendstring feed importer "$(shell date) $(shell hostname)"
	
clean:
	rm -rf $(LIB_DIR)
