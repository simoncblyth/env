#!/usr/bin/env python
"""
Status
-------

#. succeeds to mechanically login to dybsvn Trac, for automated response time measurements


"""
import mechanize, logging, os
from ConfigParser import ConfigParser 
from pprint import pformat
log = logging.getLogger(__name__)

def cnf_(path, siteconf):
    cpr=ConfigParser()
    cpr.read(os.path.expanduser(path))
    return dict(cpr.items(siteconf))

class Browser(object):
    """
    Expanding mechanize access to new sites typically requires
    ipython interactive sessions to determine how to gain automatic access.
    """
    def __init__(self, cnf ):
        br  = mechanize.Browser()
        self._basic( br, cnf )
        self._form( br, cnf )
        self.cnf = cnf 
        self.br = br

    def _basic(self, br, cnf):
        if cnf['rooturl']:
            br.add_password(cnf['rooturl'], cnf['rootuser'], cnf['rootpass'] )
        else:    
            log.warn("formlogin only")

    def _form(self, br, cnf):
        """
        Note that sometimes forms are present despite not being displayed in 
        ordinary browsers::

            In [24]: for i,f in enumerate(br.forms()):print i,f,f.method
               ....: 
            0 <GET http://dayabay.ihep.ac.cn/tracs/dybsvn/search application/x-www-form-urlencoded> GET
            1 <POST http://dayabay.ihep.ac.cn/tracs/dybsvn/login/ application/x-www-form-urlencoded
              <HiddenControl(__FORM_TOKEN=6f763fd06c6da16665cbef6c) (readonly)>
              <HiddenControl(referer=) (readonly)>
              <TextControl(user=)>
              <PasswordControl(password=)>
              <SubmitControl(<None>=Login) (readonly)>> POST

        Note that the mechanize advantage of holding the form tokens 
        """
        br.open(cnf['formurl'])
        br.select_form(nr=int(cnf['formnr']))
        f = br.form
        print f
        f[cnf['formuserkey']] = cnf['formuser']
        f[cnf['formpasskey']] = cnf['formpass']
        br.submit()
        html = br.response().read()
        print html

        links = br.links()
        #br.addheaders = [('X_REQUESTED_WITH','XMLHttpRequest')]
        for link in links:
            print link
            #br.follow_link(link)
            #html = br.response().read()
            #print html

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault('SITECONF','dybsvn_trac')
    siteconf = os.environ['SITECONF']
    cnf =  cnf_("~/.env.cnf", siteconf )
    print pformat(cnf)
    br = Browser( cnf )



