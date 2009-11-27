include Makefile.$(shell uname)

SRC_DIR     = .
INC_DIR     = .
LIB_DIR     = lib
DCT_DIR     = dict
LINKDEF     = LinkDef.hh
INCLUDES    = -I$(INC_DIR)

CFLAGS     += $(GLIB_CFLAGS)
LIBS       += $(GLIB_LIBS)
CFLAGS     += $(PCRE_CFLAGS)
LIBS       += $(PCRE_LIBS)

PRIV_CFLAGS = -I.
PRIV_LIBS   = -L./$(LIB_DIR) -lprivate

LIBFILE = libprivate.$(SOFIX)
OBJ_LIST = private.o
DOBJ_LIST = 

all : $(LIB_DIR)/private_val

$(LIB_DIR)/%.o : $(SRC_DIR)/%.c
	@echo "Compiling $< to $@ "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@ 


ifdef ROOTSYS
ROOTCFLAGS := $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTLIBS   := $(shell $(ROOTSYS)/bin/root-config --libs)
DOBJ_LIST += privateDict.o

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



$(LIB_DIR)/$(LIBFILE) : $(addprefix $(LIB_DIR)/, $(OBJ_LIST))  $(addprefix $(LIB_DIR)/, $(DOBJ_LIST)) 
	@echo "Making $@ from $^ "
	$(CC) $(SOFLAGS) $^ $(LIBS) -o $@

$(LIB_DIR)/private_val : $(addprefix $(LIB_DIR)/, private_val.o) $(LIB_DIR)/$(LIBFILE) 
	@echo "Making $@ from $< "
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(PRIV_CFLAGS) $(PRIV_LIBS) $< -o $@ 

.PHONY : clean test roottest

test:
	DYLD_LIBRARY_PATH=$(LIB_DIR) LD_LIBRARY_PATH=$(PCRE_LIBDIR):$(LIB_DIR):$(ROOTSYS)/lib ./lib/private_val AMQP_SERVER AMQP_NONEXISTING AMQP_PORT AMQP_PORTX
	
roottest:
	root -l -q tests/test_rootlookup.C


clean:
	rm -rf $(LIB_DIR) $(DCT_DIR)
