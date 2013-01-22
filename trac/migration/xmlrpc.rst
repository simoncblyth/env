XMLRPC
=======

* http://trac-hacks.org/wiki/XmlRpcPlugin

Hmm need to try to get mechanize form logins followed by xmlrpc access to work.
start trying this in `env/web/xmlrpc.py`::

    simon:migration blyth$ ./xmlrpc-test.py http://localhost/tracs/workflow/xmlrpc/
    http://localhost/tracs/workflow/xmlrpc/
    Traceback (most recent call last):
      File "./xmlrpc-test.py", line 28, in <module>
        print server.system.getAPIVersion()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/xmlrpclib.py", line 1147, in __call__
        return self.__send(self.__name, args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/xmlrpclib.py", line 1437, in __request
        verbose=self.__verbose
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/xmlrpclib.py", line 1191, in request
        headers
    xmlrpclib.ProtocolError: <ProtocolError for localhost/tracs/workflow/xmlrpc/: 401 Authorization Required>
    simon:migration blyth$ 


BUT, maybe I have jumped the authentication hurdle before, in `env/bin/tracwikidump.py`::

    simon:migration blyth$ mdfind TRAC_ENV_XMLRPC
    /Users/blyth/env/trac/migration/xmlrpc-wiki-restore.py
    /Users/blyth/env/trac/migration/xmlrpc-wiki-backup.py
    /Users/blyth/env/trac/migration/xmlrpc-test.py
    /Users/blyth/env/bin/tracwikidump.py






