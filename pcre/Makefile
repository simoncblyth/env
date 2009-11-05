include Makefile.$(shell uname)

SRC_DIR     = .
INC_DIR     = .
LIB_DIR     = lib
INCLUDES    = -I$(INC_DIR)

CFLAGS     += $(GLIB_CFLAGS)
LIBS       += $(GLIB_LIBS)
CFLAGS     += $(PCRE_CFLAGS)
LIBS       += $(PCRE_LIBS)

PRIV_CFLAGS = -I.
PRIV_LIBS   = -L./$(LIB_DIR) -lprivate

LIBFILE = libprivate.$(SOFIX)
OBJ_LIST = private.o

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 

all : $(LIB_DIR)/private_val

$(LIB_DIR)/$(LIBFILE) : $(addprefix $(LIB_DIR)/, $(OBJ_LIST)) 
	@echo "Making $@ from $^ "
	$(CC) $(SOFLAGS) $^ $(LIBS) -o $@

$(LIB_DIR)/private_val : $(addprefix $(LIB_DIR)/, private_val.o) $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(PRIV_CFLAGS) $(PRIV_LIBS) $< -o $@ 

.PHONY : clean test

test:
	DYLD_LIBRARY_PATH=$(LIB_DIR) LD_LIBRARY_PATH=$(LIB_DIR) ./lib/private_val AMQP_SERVER
	
clean:
	rm -rf $(LIB_DIR)
