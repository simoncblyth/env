# make CLHEP_BASE_DIR=$(nuwa-clhep-idir) G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$(nuwa-xercesc-idir) G4INSTALL=$(nuwa-g4-bdir)  CPPVERBOSE=1

name := xdae
G4TARGET := $(name)
#G4EXLIB := true

CPPFLAGS += -DG4VIS_USE_DAEFILE 

ifndef G4INSTALL
  G4INSTALL := $(shell bash -lc "nuwa-;nuwa-g4-bdir")
endif

CPPFLAGS += -I../DAE/include
EXTRALIBS := -lG4DAE

.PHONY: all 
all: bin 
ifndef XERCESCROOT  
	@echo XERCESCROOT not defined!
endif

include $(G4INSTALL)/config/binmake.gmk

