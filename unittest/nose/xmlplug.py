#!/usr/bin/env python
"""
Provides test results in simple XML with full stack traces and source code of failing tests.

  Get help :
       python xmlplug.py -h     
       
  Invoke a single test :
       python xmlplug.py ../../roman/roman_test.py:FromRomanBadInput.testTooManyRepeatedNumerals --with-xml-output


  NB 
      Once developments of this plugin stabilise it will be installed into NOSE_HOME allowing usage with the 
      standard nosetests entry point  

  TODO :        
         - avoid absolute paths in the output 
         - access the source for a method or function rather than the whole test class
    

"""
import sys
import os
import re
import traceback
import logging
from optparse import OptionGroup

import inspect

from nose.plugins import Plugin
from nose.inspector import inspect_traceback
#from nose.plugins.plugintest import run_buffered as run
from nose.core import run


logging.basicConfig(level=logging.INFO)
log =  logging.getLogger(__name__)



def cdata(s):
    return "<![CDATA[%s]]>" % s


def tb_filename(tb):
    return tb.tb_frame.f_code.co_filename

def tb_iter(tb):
    while tb is not None:
        yield tb
        tb = tb.tb_next

def classify_ctx( ctx ):
    if inspect.ismodule(ctx):
        return "module"
    elif inspect.isclass(ctx):
        return "class"
    elif inspect.ismethod(ctx):
        return "method"
    elif inspect.isfunction(ctx):
        return "function"
    elif inspect.iscode(ctx):
        return "code"
    elif inspect.isbuiltin(ctx):
        return "builtin"
    elif inspect.isroutine(ctx):
        return "routine"
    else:
        return "unknown"


emailpatn = re.compile(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}')
genpatn = re.compile(r'(\w*)(\(.*\))')

class Stack:
    def __init__(self):
        self.ctxs = []

    def push(self, ctx):
        self.ctxs.append(ctx)

    def pop(self):
        return self.ctxs.pop()
    
    def ctx(self,lev=-1):
        try:
            c = self.ctxs[lev]
        except IndexError,ie:
            log.error("failed to get the context for " )
            c = None
        return c

    def name(self,lev=-1):
        _ctx = self.ctx(lev)
        try:
            n = _ctx.__name__
        except AttributeError:
            n = str(_ctx).replace('<', '').replace('>', '')
        return n
    
    def file(self,lev=-1):
        _ctx = self.ctx(lev)
        try:
            path = _ctx.__file__.replace('.pyc', '.py')
        except AttributeError:
            path = ""
        return path
          
    def callable(self,name,lev=-1):
        """ is the name is of the form hello(1,2) then assume a generator and call again with hello """
        gen=genpatn.match(name)
        if gen!=None:
            log.info("callable name looks like a genrator %s " % name )
            base = gen.group(1)
            return self.callable(base,lev)
        else:
            _ctx = self.ctx(lev)
            try:
                c = _ctx.__dict__[name]
            except Exception,ev:
                print "failed to get callable for %s %s at level %d " % ( name , ev , lev )
                c = None
        return c
             
    def xml_responsible(self, name ):
        """ the test and stack context doc strings are traversed backwards, 
            the first level to yield contacts becomes the victims """
        callable = self.callable(name)
        docs = []
        if callable != None:
            docs.append(callable.__doc__)
        nctx = len(self.ctxs)
        for ic in range(nctx-1,-1,-1):
            docs.append( self.doc(ic) )
        
        victim=[]
        y=[]
        for i in range(len(docs)):
            if docs[i]==None:
                d = ""
            else:
                d=docs[i]
            resp = emailpatn.findall(d)
            if len(resp)>0 and len(victim)==0:
                victim = resp
            y.append("<doc n=\"%d\" resp=\"%s\" >%s</doc>" % ( i, " ".join(resp),  d ) ) 
        
        x=["<responsible victim=\"%s\" >" % " ".join(victim)  ] 
        x.extend(y)
        x.append("</responsible>")
        return x
    
    def doc(self, lev=-1):
        _ctx = self.ctx(lev)
        try:
            d = _ctx.__doc__
        except Exception,ev :
            print "failed to get doc %s " % ev
            d = None
        return d
          
    def dump(self):
        x=[]
        x.append("<dump>")
        for i in range(len(self.ctxs)):
            x.extend(self.xml_context(i))
        x.append("</dump>")
        return x                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                              
    def xml_context(self,lev=-1):
        _ctx = self.ctxs[lev]
        x=[]
        x.append("<context name=\"%s\"  type=\"%s\" />" %  ( self.name(lev) , classify_ctx(_ctx) ))
        #x.append( "<doc>%s</doc>" % cdata(_ctx.__doc__) )
        #x.append("</context>")
        return x
        
 


class XmlOutput(Plugin):
    """Output test results as XML
    """
    name = 'xml-output'
    score = 2 # run late
    def __init__(self):
        super(XmlOutput, self).__init__()
        self.xml = ['<report>']
        self.path = None
        self.stack = Stack()

    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)
        group = OptionGroup(parser, "%s plugin " % self.__class__.name,  "Output test results in XML with full stack traces and source code of failing tests. "  )
        group.add_option("--xml-outfile"   , default=os.environ.get('NOSE_XML_OUTFILE')  , type="string" , help="path to write xml test results to, rather than stdout, default:[%default] " ) 
        group.add_option("--xml-basepath"  , default=os.environ.get('NOSE_XML_BASEPATH')  , type="string" , help="absolute base path to be removed from paths reported in the output, default:[%default]  ")  
        group.add_option("--xml-baseprefix", default=os.environ.get('NOSE_XML_BASEPREFIX')  , type="string" , help="replace the basepath specified in --xml-basepath with this prefix, default:[%default] ")
        parser.add_option_group(group)
        
    def configure(self, options, config):
        log.debug("XmlOut configure")
        Plugin.configure(self, options, config)
        self.config = config
        self.options = options        

    def setOutputStream(self, stream):
        self.stream = stream        
        class dummy:
            def write(self, *arg):
                pass
            def writeln(self, *arg):
                pass
        d = dummy()
        return d
        
    def finalize(self , result ):
        self.xml.extend( self.xml_result( result ) )
        self.xml.append('</report>')
        if self.options.xml_outfile!=None:
            self.write_out( self.options.xml_outfile )
        else:
            for l in self.xml:
                self.stream.writeln(l)
        
    def write_out( self , path ):
        try:
            dir = os.path.dirname( path )
            if not(os.path.exists(dir)):
                os.makedirs( dir )
            f = open(path,"w")
            for l in self.xml:
                f.write(l)
                f.write("\n")  
        except IOError, (errno, strerror):
            print "I/O error(%s): %s" % (errno, strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        else:
            f.close()
    def present_path( self , path ):
        base = self.options.xml_basepath
        pfx = self.options.xml_baseprefix
        if base!=None:
            if pfx!=None:
                nath = path.replace(base,pfx)
            else:
                nath = path.replace(os.path.join(base,"/"),"")
        else:
            nath = path
        return nath
                
    def startContext( self, ctx ):
        self.stack.push(ctx)
    def stopContext(self, ctx ):
        self.stack.pop() 
    def startTest(self,test):
        pass
    def stopTest(self,test):
        pass
    def addSuccess(self,test):
        self.xml_test( test , "SUCCESS"  )
    def addError(self,test,err):
        self.xml_test( test , "ERROR" , err )
    def addFailure(self,test,err):
        self.xml_test( test , "FAIL" , err )
    def xml_test(self, test , outcome , err=None ):        
        tid = test.id()
        name = tid.split(".")[-1]
        callable = self.stack.callable(name)
        
        
        
        path, module, call = test.address()
        path = path.replace('.pyc','.py') 
        x = []
        x.append("<test name=\"%s\" outcome=\"%s\" id=\"%s\" >" % ( name , outcome , tid ))
        
        x.extend(self.stack.dump())
        x.extend(self.stack.xml_responsible(name))
        
        y = []
        if err==None:
            xml, offset = self.xml_source( callable , [] )
            y.extend(xml)
        else:
            xml, highlight = self.xml_stack( err )
            x.extend( xml )
            xml, offset = self.xml_source( callable , highlight )
            y.extend( xml )
            y.extend( self.xml_traceback( err ))
            y.extend( self.xml_detailed( err ))
        
        ppath = self.present_path(path)
        
        href = "%s#L%d" % ( ppath , offset )
        x.append("<address path=\"%s\" offset=\"%d\" module=\"%s\" call=\"%s\"  href=\"%s\" />" % ( ppath , offset , module , call, href ) )
        x.append("<doc>%s</doc>" % cdata( callable.__doc__ ) )
        
        z = ["</test>"]
        
        self.xml.extend(x)
        self.xml.extend(y)
        self.xml.extend(z)


    

    def xml_result(self, result):
        x = []
        if not result.wasSuccessful():
            conc = "FAIL"
        else:
            conc = "OK"
        x.append('<result tests=\"%d\" failures=\"%d\" errors=\"%d\" >%s</result> ' %  ( result.testsRun, len(result.failures), len(result.errors),conc ))
        return x

    def xml_source( self , obj , highlight ):
        """ gets the source for the test class ... hmm what happens with functions/generators ?? """
        x=[]
        if obj==None:
            return x, -1
        lines, offset = inspect.getsourcelines( obj   )
        x.append('<source highlight=\"%s\" >' % (  " ".join([str(h) for h in highlight ])   ) )
        for i in range(len(lines)):
            line = lines[i]
            if line and line[-1] == '\n':
                line = line[:-1]
            if offset + i in highlight:
                mark = 1
            else:
                mark = 0
            x.append('<line n=\"%d\" mark=\"%d\" ><![CDATA[%s]]></line>' % ( offset + i , mark ,  line )  )
        x.append('</source>')
        return x, offset

    def xml_stack( self , err  ):
        x=[]
        if err==None:
            return x
        ec, ev, tb = err
        x.append('<stack>')
        highlight = []
        for tb in tb_iter(tb):
            tbf = tb_filename(tb)
            tbl = tb.tb_lineno
            mark = 0
            if tbf == self.path: 
                highlight.append(tbl)
                mark = 1
            x.append('<call ln=\"%d\" mark=\"%d\" >%s</call>' % ( tbl, mark , self.present_path(tbf) ))
        x.append('</stack>')
        return (x, highlight) 

    def xml_detailed(self , err ):
        """ from the FailureDetail Plugin   formatFailure ... but pretty trivial, so dont try to get plugins working togther..
            Add detail from traceback inspection to error message of a failure."""
        x=[]
        if err==None:
            return x
        ec, ev, tb = err
        tbinfo = inspect_traceback(tb)
        x.append('<detailed>')
        x.append('<exception><![CDATA[%s]]></exception>' % str(ec) )
        x.append('<exvalue><![CDATA[%s]]></exvalue>' % str(ev) )
        x.append('<traceback><![CDATA[%s]]></traceback>' % tbinfo )
        x.append('</detailed>')    
        return x

    def xml_traceback(self, err):
        ec, ev, tb = err
        x=[]
        if err==None:
            return x
        x.append('<traceback><![CDATA[%s]]></traceback>' %  '\n'.join(traceback.format_exception(ec, ev, tb)))        
        return x






if '__main__'==__name__:
    run(argv=sys.argv,  plugins=[XmlOutput()])
