
CC = g++
LIBS = -ldl
.cc.o:
	$(CC) -ggdb -c $<
default:
	make testdcl
	OBJS = testdcl.o

testdcl: testdcl.o
	$(CC) -rdynamic -o testdcl testdcl.o $(LIBS)

libcircle.so:  circle.o
	g++ -shared -Wl,-soname,libcircle.so -o libcircle.so circle.o

libsquare.so:  square.o
	g++ -shared -Wl,-soname,libsquare.so -o libsquare.so square.o

all: testdcl libcircle.so libsquare.so
	
clean:
	rm -f *.so *.o testdcl

