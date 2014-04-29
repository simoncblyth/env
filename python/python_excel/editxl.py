#!/usr/bin/env python
"""

* http://www.youlikeprogramming.com/2012/03/examples-reading-excel-xls-documents-using-pythons-xlrd/


::

    ./editxl.py ~/Desktop/C302_20140429.xls 

This succeeds to save the copy .xls but heading formatting is lost 
compared to the origina (as checked from Numbers.app 3.2 (1861) )

"""
import os, sys, logging
log = logging.getLogger(__name__)
from collections import OrderedDict

import xlrd
from xlutils.copy import copy as xlutils_copy

only_ascii_ = lambda _:0 <= ord(_) < 128


class Cell(object):
    def __init__(self, sheet, irow, icol):
        self.irow = irow
        self.icol = icol
        self.value = sheet.cell_value(irow, icol) 
        self.type_ = sheet.cell_type(irow, icol) 

    asciivalue = property(lambda self:filter(only_ascii_, self.value))

    def __repr__(self):
         return "%s:(%s,%s) %s" % ( self.type_, self.irow, self.icol, self.asciivalue)

    def check(self):
        assert self.type_ in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_TEXT)




class Sheet(object):
    def __init__(self, name, sheet ):
        self.name = name
        self.sheet = sheet

    def row_of_cells(self, irow ):
        return [self.cell(irow, icol ) for icol in range(self.sheet.ncols)]

    def cell(self, irow, icol ):
        assert irow < self.sheet.nrows 
        assert icol < self.sheet.ncols 
        return Cell( self.sheet, irow, icol )

    def present_row(self, irow):
        return "\n".join(map(repr,self.row_of_cells(irow)))
 
    def __repr__(self):
        return "%-10s %-15s (%d,%d)  " % ( self.__class__.__name__, self.name, self.sheet.ncols, self.sheet.nrows )

    def rawcheck(self):
        sheet = self.sheet
        for i in range(sheet.nrows):
            row = sheet.row(i)
            assert type(row) == list
            assert len(row) == sheet.ncols
            for cell in row:
                assert type(cell) == xlrd.sheet.Cell
            pass

    def check(self):
        self.rawcheck()
        for irow in range(self.sheet.nrows):
            cells = self.row_of_cells(irow)
            assert len(cells) == self.sheet.ncols
            for cell in cells:
                cell.check()




class Book(OrderedDict):
    def __init__(self, rpath ):
        """
        # annoying the API of the copied is different from original
        """
        OrderedDict.__init__(self)
        rbook = xlrd.open_workbook(rpath)
        for name in rbook.sheet_names():
            self[name] = Sheet(name, rbook.sheet_by_name(name))
        pass
        self.rpath = rpath
        self.rbook = rbook
        self.wbook = xlutils_copy(rbook)
        

    def check(self):
        for name,sheet in self.items():
            sheet.check()
  
    def save(self, wpath):
        assert wpath != self.rpath
        log.info("saving to %s " % wpath)
        self.wbook.save(wpath)
 

if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    rpath = sys.argv[1]
    base, ext = os.path.splitext(rpath) 

    wpath = "".join([base,"_update",ext])
    if os.path.exists(wpath):
        os.remove(wpath)
    pass

    book = Book(rpath)
    book.check()      

    name = 'Journal paper'
    #name = 'Other'
    # for all sheets rows 0 and 1 are headers, content starting from row 2 

    sheet = book[name]
    print sheet.present_row(0)
    print sheet.present_row(1)
    print sheet.present_row(2)


    book.save(wpath)
    pass

