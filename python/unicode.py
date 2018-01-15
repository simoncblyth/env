#!/usr/bin/env python
#-*- coding: latin-1 -*-
"""
Without specifying the coding the below gives error


* https://docs.python.org/2.7/howto/unicode.html
* https://stackoverflow.com/questions/9942594/unicodeencodeerror-ascii-codec-cant-encode-character-u-xa0-in-position-20

Unicode data is usually converted to a particular encoding before it gets
written to disk or sent over a socket. It’s possible to do all the work
yourself: open a file, read an 8-bit string from it, and convert the string
with unicode(str, encoding). However, the manual approach is not recommended.


SQLite always stores text data as Unicode, using the Unicode encoding specified
when the database was created. The database driver itself takes care to return
the data as the Unicode string in the encoding used by your language/platform.

::

    In [19]: s = "abcdé"

    In [20]: print type(s),s
    <type 'str'> abcdé

    In [22]: u = u"abcdé"

    In [23]: print type(u),u
    <type 'unicode'> abcdé

    In [24]: print str(u)
    ---------------------------------------------------------------------------
    UnicodeEncodeError                        Traceback (most recent call last)
    <ipython-input-24-00223b2c9583> in <module>()
    ----> 1 print str(u)

    UnicodeEncodeError: 'ascii' codec can't encode character u'\xe9' in position 4: ordinal not in range(128)

    In [25]: print str(u.encode("UTF-8"))
    abcdé

"""


u = u'abcdé'
print ord(u[-1])



