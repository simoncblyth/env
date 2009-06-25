CPP  = g++
CC   = gcc
OBJ  = main.o FresnelData.o
LINKOBJ  = main.o FresnelData.o
LIBS =
BIN  = FresnelData
CFLAGS = -g
RM = rm -f

#-----------------------------------------
# ROOT flags and libs
#
CFLAGS += $(shell  $(ROOTSYS)/bin/root-config --cflags)
LIBS   += $(shell  $(ROOTSYS)/bin/root-config --libs)

.PHONY: all clean

all: FresnelData

clean:
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o "FresnelData" $(LIBS)

main.o: main.cpp
	$(CPP) -c main.cpp -o main.o $(CFLAGS)

FresnelData.o: FresnelData.cpp FresnelData.h
	$(CPP) -c FresnelData.cpp -o FresnelData.o $(CFLAGS)
