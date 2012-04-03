CC = g++
BOOST_HOME = /opt/local
NAME = extfun
PYLIBD = $(ENV_HOME)/lib/python2.5/site-packages

build:envcheck qxml 
	@echo -n

qxml:$(ENV_HOME)/bin/qxml
	@echo built C++ main $(shell ls -l $^)

pyx:$(PYLIBD)/_py$(NAME).so 
	@echo built swigged C++ python extension function $(shell ls -l $^)  


$(ENV_HOME)/bin/qxml: qxml.o config.o potools.o extfun.o extresolve.o
	$(CC)  -L$(BDBXML_HOME)/lib -ldbxml -ldb -L$(BOOST_HOME)/lib -lboost_program_options $^ -o $@

qxml.o: qxml.cc
	$(CC)  -I$(BDBXML_HOME)/include -I$(BOOST_HOME)/include -c $^ -o $@ 

config.o : config.cc
	$(CC)  -I$(BOOST_HOME)/include -c $^ -o $@ 

potools.o : potools.cc
	$(CC)  -I$(BOOST_HOME)/include -c $^ -o $@ 

$(NAME).o : $(NAME).cc $(NAME).hh 
	$(CC)  -I. -I$(BDBXML_HOME)/include -c $< -o $@

extresolve.o : extresolve.cc extresolve.hh 
	$(CC)  -I. -I$(BDBXML_HOME)/include -c $< -o $@


# testing config 

main : main.o config.o
	$(CC)  -L$(BOOST_HOME)/lib -lboost_program_options $^ -o $@

main.o : main.cc
	$(CC)  -I$(BOOST_HOME)/include -c $^ -o $@ 

# swig binding 

$(PYLIBD)/_py$(NAME).so : $(NAME).i
	#python setup.py build_ext --inplace
	@echo installing into $(PYLIBD)
	python setup.py install --prefix $(ENV_HOME)


.PHONY:clean pylink pyx test envcheck build qxml test_pyx

pylink:
	cd $(ENV_HOME)/bin && ln -s $(ENV_HOME)/db/bdbxml/qxml/qxml.py qxml.py 

test:
	test/extmixed.xq -l DEBUG

test_pyx:
	PYTHONPATH=$(PYLIBD) python -c "import dbxml, pyextfun "


envcheck:
ifndef BDBXML_HOME
	$(error BDBXML_HOME is not defined : use precursor bdbxml- to define it)
else
	@echo operating with BDBXML_HOME $(BDBXML_HOME)
endif
ifndef ENV_HOME
	$(error ENV_HOME is not defined : use precursor env- to define it)
endif


clean:
	rm -f *.o *.pyc *.so __db.* main && rm -rf build

