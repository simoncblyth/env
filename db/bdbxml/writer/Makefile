
CC = g++
NAME = writer

$(NAME): $(NAME).o 
	$(CC)  -L$(BDBXML_HOME)/lib -ldbxml $^ -o $@

$(NAME).o: $(NAME).cc
	$(CC) -I$(BDBXML_HOME)/include -c $^ -o $@ 


.PHONY:clean test check

test:
	./$(NAME)

check:
	dbxml -s check.dx

clean:
	rm -f *.o writer.dbxml writer

