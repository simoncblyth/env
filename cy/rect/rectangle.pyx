cdef extern from "Rectangle.h" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle(int, int, int, int)
        int x0, y0, x1, y1
        int getLength()
        int getHeight()
        int getArea()
        void move(int, int)
	
cdef class PyRectangle:
    cdef Rectangle *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, int x0, int y0, int x1, int y1):
        self.thisptr = new Rectangle(x0, y0, x1, y1)
    def __dealloc__(self):
        del self.thisptr
    def getLength(self):
        return self.thisptr.getLength()
    def getHeight(self):
        return self.thisptr.getHeight()
    def getArea(self):
        return self.thisptr.getArea()
    def move(self, dx, dy):
        self.thisptr.move(dx, dy)
