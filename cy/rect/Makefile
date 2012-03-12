.PHONY: build test clean

build:
	python setup.py build

test:
	PYTHONPATH=build/lib.macosx-10.5-ppc-2.5 python -c "from rectangle import PyRectangle ; r = PyRectangle(1,2,3,400) ; print r.getArea() "  

clean:
	rm -rf build rectangle.cpp

