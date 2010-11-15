NAME = npcy

$(NAME).so : $(NAME).pyx
	python setup.py build_ext -i && cython -a $(NAME).pyx

clean :
	rm -rf $(NAME).c $(NAME).so build

t :
	python test_$(NAME).py

i :
	ipython test_$(NAME).py


.PHONY : clean t i


