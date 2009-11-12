# === func-gen- : messaging/cjson fgp messaging/cjson.bash fgn cjson fgh messaging
cjson-src(){      echo messaging/cjson.bash ; }
cjson-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cjson-src)} ; }
cjson-vi(){       vi $(cjson-source) ; }
cjson-env(){      elocal- ; }
cjson-usage(){
  cat << EOU
     cjson-src : $(cjson-src)
     cjson-dir : $(cjson-dir)

     NB keeping makefile source in here to avoid forking cjson just to 
     add a makefile 

EOU
}
cjson-dir(){ echo $(local-base)/env/messaging/cjson ; }
cjson-cd(){  cd $(cjson-dir); }
cjson-mate(){ mate $(cjson-dir) ; }
cjson-get(){
   local dir=$(dirname $(cjson-dir)) &&  mkdir -p $dir && cd $dir
   svn co https://cjson.svn.sourceforge.net/svnroot/cjson
}

cjson-test(){
   cjson-cd
   gcc cJSON.c test.c -o test -lm
   ./test
}

cjson-name(){ echo cJSON ; }
cjson-makefile-(){  cat << EOM

include Makefile.\$(shell uname)

SRC_DIR     = .
INC_DIR     = .
LIB_DIR     = lib
INCLUDES    = -I\$(INC_DIR)

LIBFILE = lib$(cjson-name).\$(SOFIX)
OBJ_LIST = cJSON.o 

all : \$(LIB_DIR)/\$(LIBFILE)

\$(LIB_DIR)/%.o : \$(SRC_DIR)/%.c
	@echo "Compiling \$< to \$@ "
	@mkdir -p \$(LIB_DIR)
	\$(CC) \$(CFLAGS) -c \$< -o \$@ 

\$(LIB_DIR)/\$(LIBFILE) : \$(addprefix \$(LIB_DIR)/, \$(OBJ_LIST))
	@echo "Making \$@ from \$^ "
	\$(CC) \$(SOFLAGS) \$^ -o \$@

.PHONY : clean 

clean:
	rm -rf \$(LIB_DIR)

EOM
}
cjson-makefile-Darwin-(){ cat << EOM
SOFLAGS     = -dynamiclib -Wl,-undefined -Wl,dynamic_lookup
CFLAGS      = -g -fno-common -Wall
LIBS        =
SOFIX       = dylib
EOM
}
cjson-makefile-Linux-(){ cat << EOM
SOFLAGS     = -shared
CFLAGS      = -g -fno-common -Wall
LIBS        =
SOFIX       = so
EOM
}
cjson-makelib(){
  cjson-cd
  cjson-makefile- > Makefile
  cjson-makefile-Darwin-  > Makefile.Darwin
  cjson-makefile-Linux-   > Makefile.Linux
  make 
}



