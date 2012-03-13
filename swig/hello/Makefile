# http://www.swig.org/Doc1.3/Python.html

NAME = example
CC = gcc


_$(NAME).so : $(NAME)_wrap.o $(NAME).o
	$(CC)  -L/opt/local/lib -arch ppc -bundle -undefined dynamic_lookup  $^ -o $@

$(NAME).o $(NAME)_wrap.o : $(NAME)_wrap.c $(NAME).c
	$(CC) -c -I/opt/local/include/python2.5 $^ 

$(NAME)_wrap.c $(NAME).py : $(NAME).c $(NAME).i
	swig -python $(NAME).i

.PHONY: clean test


test:
	python -c "import $(NAME) as _ ; print _.fact(5),_.my_mod(15,10),_.cvar.My_variable  "

clean:
	rm -f $(NAME)_wrap.{c,o} $(NAME).py{,c} $(NAME).o _$(NAME).so 

