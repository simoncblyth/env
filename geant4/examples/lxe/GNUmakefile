# $Id: GNUmakefile,v 1.2 2004/06/01 07:05:03 gcosmo Exp $
# --------------------------------------------------------------
# GNUmakefile for LXe example.
# --------------------------------------------------------------

name := LXe
G4TARGET := $(name)
G4EXLIB := true

ifndef G4INSTALL
  G4INSTALL = ../../../..
endif

.PHONY: all
all: lib bin

include $(G4INSTALL)/config/architecture.gmk

include $(G4INSTALL)/config/binmake.gmk


# for ROOT TObject that is used by ChromaPhotonList.hh for serialisation 
INCFLAGS += -I$(ROOTSYS)/include
LDFLAGS += -L$(ROOTSYS)/lib -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -lm -ldl -rdynamic


INCFLAGS += -I$(ZEROMQ_PREFIX)/include
LDFLAGS += -L$(ZEROMQ_PREFIX)/lib -lzmq


LDFLAGS += -L$(XERCESC_PREFIX)/lib -lxerces-c


# hmm libGDML.so was compiled against the system xercesc 2_7, 
# opening can of worms with -L/usr/lib -lxercec-c fails to work 
# (maybe because the 2_8 lib of that name was already loaded)
# but explicitly providing the path to system 2_7 works
#
LDFLAGS += /usr/lib/libxerces-c.so.27
