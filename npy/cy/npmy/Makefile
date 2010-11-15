NAME = npmy

$(NAME).so : $(NAME).pyx _mysql.pxd mysql.pxd
	python setup.py build_ext -i && cython -a $(NAME).pyx

clean :
	rm -rf $(NAME).c $(NAME).so $(NAME).html build

t :
	python test_$(NAME).py

i :
	ipython test_$(NAME).py


.PHONY : clean t i


