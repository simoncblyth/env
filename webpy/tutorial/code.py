#!/usr/bin/env python
"""

::

    simon:tutorial blyth$ curl http://localhost:8080/  
    Hello, world!simon:tutorial blyth$ 


"""

import web

urls = ( '/', 'index', )

class index:
    def GET(self):
        return "Hello, world!"

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
