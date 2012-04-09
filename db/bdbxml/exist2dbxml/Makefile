
CC = g++
NAME = exist2dbxml
BOOST_HOME = /opt/local
EXE =  $(ENV_HOME)/bin/$(NAME)

all : $(EXE) clean
	ls -l $(EXE)

$(EXE) : $(NAME).o   
	$(CC)  -L$(BDBXML_HOME)/lib -ldbxml -L$(BOOST_HOME)/lib -lboost_filesystem -lboost_system  $(NAME).o -o $@

$(NAME).o: $(NAME).cc
	$(CC) -I$(BDBXML_HOME)/include -I$(BOOST_HOME)/include -c $^ -o $@ 

.PHONY:clean test all

test:
	$(NAME)

clean:
	rm -f *.o 

