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


     cjson-get
              from sourceforge svn, which annoyingly uses https: causing  "svn: SSL is not supported" 
              depending on svn build settings such as on grid1 with /disk/d4/dayabay/local/svn/subversion-1.4.0/bin/svn
              however the ancient system svn worked 
                    /usr/bin/svn co https://cjson.svn.sourceforge.net/svnroot/cjson


     cjson-makelib
              generate Makefile and use to to create dynamic lib 

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

ifdef ROOTSYS
ROOTCFLAGS := \$(shell  \$(ROOTSYS)/bin/root-config --cflags)
ROOTLIBS   := \$(shell  \$(ROOTSYS)/bin/root-config --libs)
else
missroot:
        @echo "...";
        @echo "Missing definition of environment variable 'ROOTSYS' !";
        @echo "...";
endif

SRC_DIR     = .
INC_DIR     = .
LIB_DIR     = lib
DCT_DIR     = dict
INCLUDES    = -I\$(INC_DIR)
LINKDEF     = LinkDef.hh

LIBFILE = lib$(cjson-name).\$(SOFIX)
OBJ_LIST = cJSON.o 
DOBJ_LIST = cJSONDict.o

CFLAGS     += \$(ROOTCFLAGS)
LIBS       += \$(ROOTLIBS)

all : \$(LIB_DIR)/\$(LIBFILE)

\$(LIB_DIR)/%.o : \$(SRC_DIR)/%.c
	@echo "Compiling \$< to \$@ "
	@mkdir -p \$(LIB_DIR)
	\$(CC) \$(CFLAGS) -c \$< -o \$@ 

\$(LIB_DIR)/%.o : \$(DCT_DIR)/%.cxx
	@echo "Compiling \$< to \$@ "
	@mkdir -p \$(LIB_DIR)
	\$(CC) \$(CFLAGS) -c \$< -o \$@ -I.

\$(DCT_DIR)/%Dict.cxx : \$(INC_DIR)/%.h \$(INC_DIR)/%_\$(LINKDEF)
	@echo "Creating \$@ from $^ "
	@mkdir -p \$(DCT_DIR)
	\$(ROOTSYS)/bin/rootcint -f \$@ -c \$(INCLUDES) $^


\$(LIB_DIR)/\$(LIBFILE) : \$(addprefix \$(LIB_DIR)/, \$(OBJ_LIST)) \$(addprefix \$(LIB_DIR)/, \$(DOBJ_LIST))
	@echo "Making \$@ from \$^ "
	\$(CC) \$(SOFLAGS) \$^ -o \$@

.PHONY : clean 

clean:
	rm -rf \$(LIB_DIR) \$(DCT_DIR)

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
  cjson-linkdef-  > cJSON_LinkDef.hh
  cjson-test- > $(cjson-testname)
  make 
  cjson-test
}

cjson-clean(){
  cjson-cd
  [ -f Makefile ] && make clean
  rm -rf Makefile Makefile.Darwin Makefile.Linux cJSON_LinkDef.hh $(cjson-testname)
}


cjson-linkdef-(){ cat << EOL
// from $FUNCNAME  gleaned by examination of TSystem::CompileMacro and running this in debug, see wiki:MidasMQ
#ifdef __CINT__ 

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedefs;
#pragma link C++ defined_in ./cJSON.h;
//#pragma link C++ defined_in ./cJSON.c;

#endif  

EOL
}

cjson-test-(){ cat << EOT
{
   gSystem->Load(Form("$LOCAL_BASE/env/messaging/cjson/lib/libcJSON.%s",gSystem->GetSoExt()));
   cJSON* root =  cJSON_CreateObject()  ; 
   cJSON_AddItemToObject(root,"number", cJSON_CreateString("hello world") );
   cout << cJSON_Print(root) << endl ;
}
EOT
}

cjson-testname(){ echo test_rootcjson.C ; }
cjson-test(){ root -l -q $(cjson-testname) ; }
