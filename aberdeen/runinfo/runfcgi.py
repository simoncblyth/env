#!/usr/bin/env python
##  http://just-another.net/2009/01/18/byteflowdjangosupervisordnginx-win/
if __name__ == '__main__':
    from flup.server.fcgi_fork import WSGIServer
    from django.core.handlers.wsgi import WSGIHandler
    WSGIServer(WSGIHandler()).run()

