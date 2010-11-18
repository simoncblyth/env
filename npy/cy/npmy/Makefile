NAME = npmy

$(NAME).so : $(NAME).pyx _mysql.pxd mysql.pxd
	python setup.py build_ext -i && cython -a $(NAME).pyx

clean :
	rm -rf $(NAME).c $(NAME).so $(NAME).html build *.pyc

t0 :
	python test_$(NAME).py 0
t1 :
	python test_$(NAME).py 1
t2 :
	python test_$(NAME).py 2

t :
	python test_$(NAME).py 

ts :
	python test_$(NAME).py 0
	python test_$(NAME).py 1
	python test_$(NAME).py 2

i :
	ipython test_$(NAME).py

p :
	ipython plt.py


.PHONY : clean t0 t1 t2 t ts i p


