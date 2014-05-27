CPPFLAGS+=-I./include -I$(shell root-config --incdir) 
CPPFLAGS+=$(shell pkg-config --cflags libzmq) -DWITH_ZMQ

LDFLAGS+=$(shell root-config --libs)
LDFLAGS+=$(shell pkg-config --libs libzmq)

all: libs

libs: libZMQRoot.so

libZMQRoot.so: src/MyTMessage.o src/ZMQRoot.o src/MyTMessageDict.o
	g++ -fPIC -shared -o $@ $? ${LDFLAGS}

src/MyTMessage.o: src/MyTMessage.cc
	g++ -fPIC -c -o $@ $< ${CPPFLAGS}

src/MyTMessageDict.o: include/MyTMessage.hh include/MyTMessage_LinkDef.h
	rootcint -f src/MyTMessageDict.cc -c -p ${CPPFLAGS} include/MyTMessage.hh include/MyTMessage_LinkDef.h
	g++ -fPIC -c -o $@ src/MyTMessageDict.cc ${CPPFLAGS}

src/ZMQRoot.o: src/ZMQRoot.cc
	g++ -fPIC -c -o $@ $< ${CPPFLAGS}

tests/client: tests/client.cc libZMQRoot.so
	g++ -fPIC -Wl,-rpath=. -o $@ $< ${CPPFLAGS} ${LDFLAGS} -L. -lZMQRoot
tests/worker: tests/worker.cc libZMQRoot.so
	g++ -fPIC -Wl,-rpath=. -o $@ $< ${CPPFLAGS} ${LDFLAGS} -L. -lZMQRoot
