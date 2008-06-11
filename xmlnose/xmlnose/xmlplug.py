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

import logging
from optparse import OptionGroup

from nose.plugins import Plugin
from nose.inspector import inspect_traceback
from nose.core import run

from stack import Stack, tb_iter, tb_filename, exception_lines, source_range 
import inspect
import traceback
import time

log =  logging.getLogger(__name__)
log.setLevel(logging.INFO)


def cdata(s):
    return "<![CDATA[%s]]>" % s


class XmlOutput(Plugin):
    """Output test results as XML
    """
    name = 'xml-output'
    score = 2 # run late
    def __init__(self):
        super(XmlOutput, self).__init__()
        self.xml = ['<report category="test" generator="http://bitten.cmlenz.net/tools/python#unittest"  >']
        self.path = None
        self.stack = Stack()
        self.duration_ = {}

    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)
        group = OptionGroup(parser, "%s plugin " % self.__class__.name,  "Output test results in XML with full stack traces and source code of failing tests. "  )
        group.add_option("--xml-outfile"   , default=os.environ.get('NOSE_XML_OUTFILE')  , type="string" , help="path to write xml test results to, rather than stdout, default:[%default] " ) 
        group.add_option("--xml-basepath"  , default=os.getcwd()+"/" , type="string" , help="absolute base path to be removed from paths reported in the output, defaults to invoking directory:[%default]  ")  
        group.add_option("--xml-baseprefix", default=""              , type="string" , help="replace the basepath with this prefix, default:[%default] ")
        fmts = ['debug','bitten']
        group.add_option("--xml-format"  ,   default=fmts[-1]  , choices=fmts , help="choose one of: %s   default:[%%default] " % ", ".join(fmts) )
        parser.add_option_group(group)
        
    def configure(self, options, config):
        log.debug("XmlOut configure")
        Plugin.configure(self, options, config)
        self.config = config
        self.options = options        

    def setOutputStream(self, stream):
        """Called before test output begins. 
          To direct test output to a new stream, return a stream object,
          which must implement a write(msg) method. If you only want to note the stream, 
          not capture or redirect it, then return None.
        """
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
            outpath = os.path.abspath(self.options.xml_outfile)
            self.write_out( outpath )
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
        test.test._starttime = time.time()
        #print "startTest %s %s %s " % ( test , test.test , test.test._starttime )
    def stopTest(self,test):
        """ this is called after the test outcome is reported ... so doing duration calc here misses the boat """
        pass
        
    def addSuccess(self,test):
        self.xml_test( test , "success"  )
    def addError(self,test,err):
        self.xml_test( test , "error" , err  )
    def addFailure(self,test,err):
        self.xml_test( test , "failure" , err  )
    def xml_test(self, test , status , err=None ):        
        """
       
        Adopt the bitten format 
             http://bitten.edgewall.org/wiki/ReportFormats
             http://groups.google.com/group/bitten/browse_thread/thread/537d240d0124cde0 
          
        duration: float duration of test 
          status: string "success" or "failure" or "error" 
            name: string name of the test 
         fixture: string name of the test fixture 
            file: path to test file relative to the base path for the build configuration 
          stdout: The output from the test 
       traceback: The traceback from any error or failure. 
       
       examples...
         fixture="bitten.tests.model.BuildConfigTestCase"
         name="test_config_update_name" 
         file="bitten/tests/model.py"
        
        http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/unittest/demo/Users/blyth/env/unittest/demo/package/module_test.py
        http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/unittest/demo/Users/blyth/env/unittest/demo/package/subpkg/submod_test.py
        
        the file should be relative to 
                              /Users/blyth/env/unittest/demo
                 
            the path for the config is   trunk/unittest/demo     
                 
        """
        
        dbg = self.options.xml_format=='debug'
        
        tid = test.id()
        name = tid.split(".")[-1]
        callable = self.stack.callable(name)
        
        dur =  time.time() - test.test._starttime 
               
        path, module, call = test.address()
        path = path.replace('.pyc','.py') 
        x = []
        x.append("<test>")
        x.append("<name>%s</name>" % name )
        x.append("<fixture>%s</fixture>" % tid )
        x.append("<duration>%f</duration>" %  dur )        
        x.append("<status>%s</status>" % status ) 
        x.append("<file>%s</file>" % self.present_path(path) )
        x.append("<range>%s</range>" % source_range( callable ))
        if  err!=None:      
            x.append("<stdout><![CDATA[%s]]></stdout>" % '\n'.join(traceback.format_exception(*err))  )        
            x.append("<lines>%s</lines>" %  " ".join([str(l) for l in exception_lines(err,path)]))
       
        x.append("</test>")
        self.xml.extend(x)

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


 
    
    def xml_detailed(self , err ):
        """ from the FailureDetail Plugin   formatFailure ... but pretty trivial, so dont try to get plugins working togther..
            Add detail from traceback inspection to error message of a failure.
        
            provides a little context around the failing line... but does not give a full traceback 
        """
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

 




if '__main__'==__name__:
    run(argv=sys.argv,  plugins=[XmlOutput()])
