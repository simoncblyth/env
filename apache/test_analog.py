
from env.apache.analog import load
from StringIO import StringIO

orig = StringIO(r"""207.46.13.131 - - [17/Jan/2011:11:48:10 +0800] "GET /tracs/env/ticket/228 HTTP/1.1" 200 14270
77.88.27.27 - - [17/Jan/2011:11:48:20 +0800] "GET /tracs/env/export/3255/trunk/vdbi/vdbi/tw/jquery/__init__.py HTTP/1.1" 200 -
77.88.27.27 - - [17/Jan/2011:11:48:29 +0800] "GET /tracs/env/changeset/43 HTTP/1.1" 200 9764
77.88.27.27 - - [17/Jan/2011:11:48:41 +0800] "GET /tracs/env/changeset/1908/trunk/nuwa HTTP/1.1" 200 15076
77.88.27.27 - - [17/Jan/2011:11:49:14 +0800] "GET /tracs/env/wiki/ApacheBenchmarks HTTP/1.1" 200 51613
77.88.27.27 - - [17/Jan/2011:11:49:26 +0800] "GET /tracs/tracdev/changeset/55/trac2mediawiki/trunk/0.11/plugins/trac2mediawiki/__init__.py HTTP/1.1" 200 10794
77.88.27.27 - - [17/Jan/2011:11:49:36 +0800] "GET /repos/env/trunk/dyw/root_use.bash HTTP/1.1" 200 1821
77.88.27.27 - - [17/Jan/2011:11:49:44 +0800] "GET /tracs/env/changeset/2422/trunk/mysql HTTP/1.1" 200 10162
77.88.27.27 - - [17/Jan/2011:11:49:58 +0800] "GET /tracs/env/changeset/2823/trunk/#hello HTTP/1.1" 200 13307
77.88.27.27 - - [17/Jan/2011:11:50:09 +0800] "GET /tracs/env/changeset/2082/trunk/thho HTTP/1.1" 200 6551
""")

corr = StringIO(r"""65.55.211.22 - - [04 May 2009 12:44:30 +0800] "GET /tracs/env/wiki/Athena?action=diff&version=15 HTTP/1.1" 200 9384
  66.249.72.201 - - [04 May 2009 12:44:58 +0800] "GET /tracs/env/browser/trunk/test/run.py?rev=1410 HTTP/1.1" 200 30021
  65.55.211.22 - - [04 May 2009 12:45:02 +0800] "GET /tracs/env/wiki/VGM?action=diff&version=2 HTTP/1.1" 200 9426
  140.112.101.191 - - [04 May 2009 12:45:02 +0800] "OPTIONS /repos/env/trunk HTTP/1.1" 200 189
  140.112.101.191 - - [04 May 2009 12:45:02 +0800] "MKACTIVITY /repos/env/!svn/act/e90552d3-0e69-0410-bc7e-b5fb481100c6 HTTP/1.1" 401 534
  65.55.211.22 - - [04 May 2009 12:45:03 +0800] "GET /tracs/env/wiki/ThhoPython?action=diff&version=12 HTTP/1.1" 200 9216
  65.55.211.22 - - [04 May 2009 12:45:08 +0800] "GET /tracs/env/wiki/ThhoPython?action=diff&version=11 HTTP/1.1" 200 20552
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "MKACTIVITY /repos/env/!svn/act/e90552d3-0e69-0410-bc7e-b5fb481100c6 HTTP/1.1" 201 363
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/trunk HTTP/1.1" 207 440
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/!svn/vcc/default HTTP/1.1" 207 403
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "CHECKOUT /repos/env/!svn/bln/1962 HTTP/1.1" 201 380
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPPATCH /repos/env/!svn/wbl/e90552d3-0e69-0410-bc7e-b5fb481100c6/1962 HTTP/1.1" 207 357
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/trunk HTTP/1.1" 207 399
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/trunk/env.bash HTTP/1.1" 207 672
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/!svn/vcc/default HTTP/1.1" 207 460
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/!svn/bc/1952/trunk/env.bash HTTP/1.1" 207 429
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "CHECKOUT /repos/env/!svn/ver/1945/trunk/env.bash HTTP/1.1" 201 390
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/trunk/scm HTTP/1.1" 207 407
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/trunk/scm/scm-backup.bash HTTP/1.1" 207 694
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/!svn/vcc/default HTTP/1.1" 207 460
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PROPFIND /repos/env/!svn/bc/1952/trunk/scm/scm-backup.bash HTTP/1.1" 207 451
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "CHECKOUT /repos/env/!svn/ver/1950/trunk/scm/scm-backup.bash HTTP/1.1" 201 401
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PUT /repos/env/!svn/wrk/e90552d3-0e69-0410-bc7e-b5fb481100c6/trunk/scm/scm-backup.bash HTTP/1.1" 204 -
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "PUT /repos/env/!svn/wrk/e90552d3-0e69-0410-bc7e-b5fb481100c6/trunk/env.bash HTTP/1.1" 204 -
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "MERGE /repos/env/trunk HTTP/1.1" 200 1014
  140.112.101.191 - blyth [04 May 2009 12:45:09 +0800] "DELETE /repos/env/!svn/act/e90552d3-0e69-0410-bc7e-b5fb481100c6 HTTP/1.1" 204 -
""")

a = load( corr )


