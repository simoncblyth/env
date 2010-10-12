

import re 
import logging
import traceback
import inspect

log =  logging.getLogger(__name__)
log.setLevel(logging.INFO)


def tb_filename(tb):
    return tb.tb_frame.f_code.co_filename

def tb_iter(tb):
    while tb is not None:
        yield tb
        tb = tb.tb_next

def exception_lines(err , path ):
    """ examine the traceback to obtain list of lines that are in path """
    x=[]
    if err==None:
        return x
    ec, ev, tb = err
    for tb in tb_iter(tb):
        tbf = tb_filename(tb)
        tbl = tb.tb_lineno
        if tbf == path: 
            x.append(tbl)
    return x 

   
def source_range( callable ):
    """ inspect the source code of the callable to determine unity based line range in the file  """
    if not callable:
        return "0-0" 
    lines, offset = inspect.getsourcelines( callable )
    return "%d-%d" % ( offset , offset+len(lines) )





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
            log.debug("callable name looks like a genrator %s " % name )
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
        
 
