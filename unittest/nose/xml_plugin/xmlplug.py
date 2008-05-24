"""
Based on the html_plugin example
"""
import sys
import traceback
from inspect import getsourcelines
from nose.plugins import Plugin
from nose.inspector import inspect_traceback
from nose.plugins.plugintest import run_buffered as run

def tb_filename(tb):
    return tb.tb_frame.f_code.co_filename

def tb_iter(tb):
    while tb is not None:
        yield tb
        tb = tb.tb_next

class XmlOutput(Plugin):
    """Output test results as XML
       how to test : 
        python xmlplug.py ../../roman/roman_test.py:FromRomanBadInput.testTooManyRepeatedNumerals --with-xml-output
    """
    name = 'xml-output'
    score = 2 # run late
    def __init__(self):
        super(XmlOutput, self).__init__()
        self.xml = ['<report>']
        self.file = None

    def setOutputStream(self, stream):
        # grab for own use
        self.stream = stream        
        # return dummy stream
        class dummy:
            def write(self, *arg):
                pass
            def writeln(self, *arg):
                pass
        d = dummy()
        return d
        
    def finalize(self , result ):
        self.xml.append('<finalize tests=\"%d\" failures=\"%d\" errors=\"%d\" > ' %  ( result.testsRun, len(result.failures), len(result.errors) ))
        if not result.wasSuccessful():
            conc = "FAILED"
        else:
            conc = "OK"
        self.xml.append('<conclusion>%s</conclusion>' % conc )
        self.xml.append('</finalize>')
        self.xml.append('</report>')
        for l in self.xml:
            self.stream.writeln(l)
        
    def startContext( self, ctx ):
        try:
            n = ctx.__name__
        except AttributeError:
            n = str(ctx).replace('<', '').replace('>', '')
        self.xml.append("<context name=\"%s\">" % n )
        try:
            path = ctx.__file__.replace('.pyc', '.py')
            self.xml.append("<path><![CDATA[%s]]></path>" %  path ) 
        except AttributeError:
            pass

    def stopContext(self, ctx ):
        self.xml.append("</context>")
        
    def startTest(self,test):
        file, module, call = test.address()
        self.file = file
        self.xml.append('<test id=\"%s\" file=\"%s\" module=\"%s\" call=\"%s\"  >' % ( test.id() , file , module , call ) )
        tt = type(test.test)
        self.xml.append('<type><![CDATA[%s]]></type>' %  tt ) 
        self.xml.append('<description><![CDATA[%s]]></description>' % test.shortDescription() or str(test) )
    
    def stopTest(self,test):
        self.xml.append('</test>')
    
    def addSuccess(self,test):
        name = "success"
        self.xml.append("<%s id=\"%s\" >" % ( name , test.id() ) )
        self.source( test )
        self.xml.append("</%s>" % name )
        
    def addError(self,test,err):
        name = "error"
        self.xml.append("<%s id=\"%s\" >" %  ( name , test.id() ) )
        self.stack( err )
        self.source( test , self.lines )
        self.error( err )
        self.detailed( err )
        self.xml.append("</%s>" % name )
    
    def addFailure(self,test,err):
        name = "failure"
        self.xml.append("<%s id=\"%s\" >" % ( name , test.id() ) )
        self.stack( err )
        self.source( test , self.lines )
        self.error( err )
        self.detailed( err )
        self.xml.append("</%s>" % name )    
        

    def source( self , test , highlight=[] ):
        ## gets the source for the test class ... hmm what happens with functions/generators ??
        tt = type(test.test)
        lines, nline = getsourcelines( tt  )
        self.xml.append('<source id=\"%s\" >' % test.id() )
        for i in range(len(lines)):
            line = lines[i]
            if line and line[-1] == '\n':
                line = line[:-1]
            if i in highlight:
                mark = 1
            else:
                mark = 0
            self.xml.append('<line n=\"%d\" mark=\"%d\" ><![CDATA[%s]]></line>' % ( nline + i , mark ,  line )  )
        self.xml.append('</source>')

    def stack( self , err  ):
        ec, ev, tb = err
        self.xml.append('<stack>')
        self.lines = []
        for tb in tb_iter(tb):
            tbf = tb_filename(tb)
            tbl = tb.tb_lineno
            mark = 0
            if tbf == self.file: 
                self.lines.append(tbl)
                mark = 1
            self.xml.append('<call ln=\"%d\" mark=\"%d\" >%s</call>' % ( tbl, mark , tbf ))
        self.xml.append('</stack>')

    def detailed(self , err ):
        """ from the FailureDetail Plugin   formatFailure ... but pretty trivial, so dont try to get plugins working togther..
            Add detail from traceback inspection to error message of a failure."""
        ec, ev, tb = err
        tbinfo = inspect_traceback(tb)
        self.xml.append('<detailed>')
        self.xml.append('<exception><![CDATA[%s]]></exception>' % str(ec) )
        self.xml.append('<exvalue><![CDATA[%s]]></exvalue>' % str(ev) )
        self.xml.append('<traceback><![CDATA[%s]]></traceback>' % tbinfo )
        self.xml.append('</detailed>')    

    def error(self, err):
        ec, ev, tb = err
        self.xml.append('<error><![CDATA[%s]]></error>' %  '\n'.join(traceback.format_exception(ec, ev, tb)))        






if '__main__'==__name__:
    run(argv=sys.argv,  plugins=[XmlOutput()])
