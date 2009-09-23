from vdbi.app import serve_app
from optparse import OptionParser
import sys

def vdbi():
    parser = OptionParser()
    parser.add_option('', '--dburl',
                  dest='url',
                  help='SQLAlchemy database uri (eg: postgres:///somedatabase)',
                  default='sqlite:///rum_demo.db')
    parser.add_option('-d', '--debug',
                  dest='dbg',
                  help='Turn on debug mode',
                  default=False,
                  action='store_true')
    opts, args = parser.parse_args(sys.argv)
    kwa = { 'url':opts.url , 'dbg':opts.dbg }
    print "vdbi kwa %s " % ( repr(kwa) )
    app = serve_app(**kwa)
    
    
def parentage(p):
    """
       Element tree before 1.3 is amazingly braindead ... 
          * no parent setting
          * cannot find by id / class        
    """
    for c in p:
        c.parent = p
        parentage(c)

def xml_parse( txt ):
    from StringIO import StringIO
    if txt.startswith("http"):
        import urllib
        txt = urllib.urlopen(txt).read()
    demo = StringIO( str(txt) )
    from xml.etree import ElementTree as ET
    t = ET.parse( demo )
    r = t.getroot()
    parentage(r)
    return r, t 


def resource_path(path, name=None):
    n = path.find(":")
    if n>-1:
        from pkg_resources import resource_filename
        path = resource_filename(path[0:n],path[n+1:])
    import os
    path = os.path.abspath(path)
    if name:
        return os.path.join(path, name)
    return path
        
def scrape(url='http://localhost:8080/login', name='login.html'):
    import os
    cmd = "curl --user-agent 'MSIE' '%s'" % url
    print "performing %s " % cmd
    html = os.popen(cmd).read()
    print html
    path = resource_path('vdbi.rum:templates', name=name)
    print "writing scraped html to %s" % path
    file(path,"w").write(html)
    return html
    
def transfer_statics(target='plvdbi:public', name='login.html' , src="vdbi.rum.widgets:static"):
    """
       Copies over statics linked from the scraped .html 
       into the target folder 
       
       resources are specified by the pkg_resource.resource_filename arguments of
       the python package and subdirectory 
       
    """
    import os
    path = resource_path('vdbi.rum:templates', name=name)
    if not(os.path.exists(path)):
        print "no path %s ... run the command \"scrape first\" to create it " % path 
        return
    
    print "reading html from %s" % path
    html = file(path,"r").read()
    r, t = xml_parse(html)
    
    targetd = resource_path(target)
    assert os.path.isdir(targetd)
    
    srcd = resource_path(src)
    assert os.path.isdir(srcd)
    
    for link in r.findall(".//{http://www.w3.org/1999/xhtml}link"):
        href = link.attrib['href']
        if href.startswith('/toscawidgets'):
            
            href = href.lstrip('/')
            name = os.path.basename(href)
            
            dir = os.path.join(targetd, os.path.dirname(href))
            if not(os.path.exists(dir)):os.makedirs(dir)
                
            src = os.path.join( srcd, name)
            assert os.path.exists(src), "no such src path %s " % src
            
            dst = os.path.join( dir, name )
            print "copying from %s to %s" % ( src, dst )
            import shutil
            shutil.copy( src, dst )
    
    
if __name__=='__main__':
    transfer_statics()
    

            
                
    
    
    
