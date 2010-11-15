NAME = hello

$(NAME).so : $(NAME).pyx
	python setup.py build_ext -i

clean :
	rm -rf $(NAME).c $(NAME).so $(NAME).html build

t :
	python test_$(NAME).py

i :
	ipython test_$(NAME).py


.PHONY : clean t i


