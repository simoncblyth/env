CPP  = g++
CC   = gcc
OBJ  = mold_status.o
LINKOBJ  = mold_status.o
LIBS =
INCS =
BIN  = mold_status
CFLAGS = $(INCS)
RM = rm -f

#-----------------------------------------
# ROOT flags and libs
#
CFLAGS += $(shell  $(ROOTSYS)/bin/root-config --cflags)
LIBS   += $(shell  $(ROOTSYS)/bin/root-config --libs)

.PHONY: all clean

all: mold_status

clean:
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -o "mold_status" $(LIBS)

mold_status.o: mold_status.cpp
	$(CPP) -c mold_status.cpp -o mold_status.o $(CFLAGS)
