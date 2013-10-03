
name := xdae
G4TARGET := $(name)
G4EXLIB := true

ifndef G4INSTALL
	@echo G4INSTALL not defined!
endif
ifndef XERCESCROOT  
	@echo XERCESCROOT not defined!
endif

CPPFLAGS += -DG4VIS_USE_DAEFILE 
CPPFLAGS += -I./include -I../DAE/include
EXTRALIBS := -lG4DAE

.PHONY: all 
all: lib bin 

include $(G4INSTALL)/config/binmake.gmk

