"""
  
  Exploring 
       * creation of a docutils table from directly from tabledata ...
       * output as RST

  With 
       * simplification of docutils rst parser 
       * non-standard rst writer 
            svn export http://svn.berlios.de/svnroot/repos/docutils/branches/lossless-rst-writer/docutils/writers/rst.py

            THIS IS DOES NOT OUTPUT NICE RST ... SO MOVED TO converter-tabledoc.py / tabledoc.py approach 

"""
import sys
from converter import RestWriter

from docutils.core import publish_from_doctree
from docutils import nodes
from docutils.writers.rst import Writer 
from docutils.utils import Reporter


class en(object):
    def __init__(self):
        self.language_code = "en"

class rep(object):
    def __init__(self):
        self.debug = True


class TableDoc:

    def table(self, tabledata):
        settings = en()
        reporter = Reporter( "tabledata", 5,5 )
        doc = nodes.document( settings , reporter )
        t = self.build_table( tabledata , 0 )
        doc += t 
        return doc

    def build_table(self, tabledata, tableline, stub_columns=0):
        colwidths, headrows, bodyrows = tabledata
        table = nodes.table()
        tgroup = nodes.tgroup(cols=len(colwidths))
        table += tgroup
        for colwidth in colwidths:
            colspec = nodes.colspec(colwidth=colwidth)
            if stub_columns:
                colspec.attributes['stub'] = 1
                stub_columns -= 1
            tgroup += colspec
        if headrows:
            thead = nodes.thead()
            tgroup += thead
            for row in headrows:
                thead += self.build_table_row(row, tableline)
        tbody = nodes.tbody()
        tgroup += tbody
        for row in bodyrows:
            tbody += self.build_table_row(row, tableline)
        return table

    def build_table_row(self, rowdata, tableline):
        row = nodes.row()
        for cell in rowdata:
            if cell is None:
                continue
            morerows, morecols, offset, cellblock = cell
            attributes = {}
            if morerows:
                attributes['morerows'] = morerows
            if morecols:
                attributes['morecols'] = morecols
            entry = nodes.entry(**attributes)
            row += entry
            entry += nodes.paragraph( text=cellblock )
        return row



tabledata = ([24, 12, 10, 10],
         [[(0, 0, 1, ['Header row, column 1']),
           (0, 0, 1, ['Header 2']),
           (0, 0, 1, ['Header 3']),
           (0, 0, 1, ['Header 4'])]],
         [[(0, 0, 3, ['body row 1, column 1']),
           (0, 0, 3, ['column 2']),
           (0, 0, 3, ['column 3']),
           (0, 0, 3, ['column 4'])],
          [(0, 0, 5, ['body row 2']),
           (0, 2, 5, ['Cells may span columns.']),
           None,
           None],
          [(0, 0, 7, ['body row 3']),
           (1, 0, 7, ['Cells may', 'span rows.', '']),
           (1, 1, 7, ['- Table cells', '- contain', '- body elements.']),
           None],
          [(0, 0, 9, ['body row 4']), None, None, None]])



if __name__ == '__main__':

   td = TableDoc()
   doc = td.table( tabledata )
   print doc
 
   w = Writer()
   p = publish_from_doctree( doc)

   print p

   #o = open("tabledoc.rst","w")
   #w.write( t , o )
   #o.close() 


