# http://www.swig.org/Doc1.3/Python.html

NAME = Rectangle
CC = g++
PYINC = /opt/local/include/python2.5

_$(NAME).so : $(NAME)_wrap.o $(NAME).o
	$(CC)  -L/opt/local/lib -arch ppc -bundle -undefined dynamic_lookup  $^ -o $@

# cleaner to split
#$(NAME).o $(NAME)_wrap.o : $(NAME)_wrap.cxx $(NAME).cpp
#	$(CC) -c -I/opt/local/include/python2.5 $^ 

$(NAME)_wrap.o : $(NAME)_wrap.cxx 
	$(CC) -c -I$(PYINC) $^ 

$(NAME).o : $(NAME).cpp
	$(CC) -c  $^ 

$(NAME)_wrap.cxx $(NAME).py : $(NAME).cpp $(NAME).i
	swig -c++ -python $(NAME).i

.PHONY: clean test


test:
	python -c "import Rectangle as _ ; r = _.Rectangle(1,2,3,4) ; print r.getArea() "

clean:
	rm -f $(NAME)_wrap.{cxx,o} $(NAME).py{,c} $(NAME).o _$(NAME).so 

