NAME = test_recarray

$(NAME).so : $(NAME).pyx
	python setup.py build_ext -i

clean :
	rm -rf $(NAME).c $(NAME).so build

t :
	python test.py

i :
	ipython test.py


.PHONY : clean t i


