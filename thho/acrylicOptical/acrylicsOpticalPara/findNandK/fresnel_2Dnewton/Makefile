CPP  = g++
CC   = gcc
OBJ  = main.o findNKNewton.o
LINKOBJ  = main.o findNKNewton.o
LIBS =
BIN  = findNKNewton
CFLAGS = -g
RM = rm -f

#-----------------------------------------
# ROOT flags and libs
#
CFLAGS += $(shell  $(ROOTSYS)/bin/root-config --cflags)
LIBS   += $(shell  $(ROOTSYS)/bin/root-config --libs)

.PHONY: all clean

all: findNKNewton

clean:
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o "findNKNewton" $(LIBS)

main.o: main.cpp
	$(CPP) -c main.cpp -o main.o $(CFLAGS)

findNKNewton.o: findNKNewton.cpp findNKNewton.h
	$(CPP) -c findNKNewton.cpp -o findNKNewton.o $(CFLAGS)
