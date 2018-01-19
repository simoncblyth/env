Python Unicode
=================

Experience
------------

Initial Observations
~~~~~~~~~~~~~~~~~~~~~~~

1. seems cannot return unicode from __str__ always returns str (bytes)
2. **overloading __str__ is bad practice when dealing with unicode**

All py2 ascii byte strings get implicitly decoded into unicode 
**assuming that they are ascii** when those str(py2) 
are combined with unicode such as ``u""``


Python2 has __unicode__ for precisely this purpose, return encoded bytes from __str__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/1307014/python-str-versus-unicode


**John Millikin:**

* __str__() is the old method -- it returns bytes
* __unicode__() is the new, preferred method -- it returns characters. 

The names are a bit confusing, but in 2.x we're stuck with them for compatibility reasons. 
Generally, you should put all your string formatting in __unicode__(), and create a stub __str__() method:

::

    def __str__(self):
        return unicode(self).encode('utf-8')

In 3.0, str contains characters, so the same methods are 
named __bytes__() and __str__(). These behave as expected.



refs
------

* http://docs.python.org/2/howto/unicode.html
* https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/
* http://nedbatchelder.com/text/unipain.html


**Joel**

* (Old UCS-2 way of storing Unicode in 2 bytes)
  So the people were forced to come up with the bizarre convention of storing a
  FE FF at the beginning of every Unicode string; this is called a Unicode Byte
  Order Mark and if you are swapping your high and low bytes it will look like a
  FF FE and the person reading your string will know that they have to swap every
  other byte

* In UTF-8, every code point from 0-127 is stored in a single byte. Only code
  points 128 and above are stored using 2, 3, in fact, up to 6 bytes.
  This has the neat side effect that English text looks exactly the same in UTF-8
  as it did in ASCII, so Americans don’t even notice anything wrong. 

* If there’s no equivalent for the Unicode code point you’re trying to represent
  in the encoding you’re trying to represent it in, you usually get a little
  question mark: ? or, if you’re really good, a box. Which did you get? -> �

::

    In [29]: c = "�"

    In [30]: print c
    �

    In [31]: print ord(c)
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-31-8c8a0a4709ff> in <module>()
    ----> 1 print ord(c)

    TypeError: ord() expected a character, but string of length 3 found

    In [32]: c = u"�"

    In [33]: print ord(c)
    65533

    In [34]: print hex(ord(c))
    0xfffd

    In [35]: print unichr(0xffff)    ## curious, this displays in ipython (white slash circle on black background) but cannot copy/paste into vim
    ￿



* UTF 7, 8, 16, and 32 all have the nice property of being able to store any code point correctly.

* **It does not make sense to have a string without knowing what encoding it uses.**

* **You can no longer stick your head in the sand and pretend that “plain” text is ASCII.**

* **There Ain’t No Such Thing As Plain Text.**


::

    <html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">


* that meta tag really has to be the very first thing in the <head> section
  because as soon as the web browser sees this tag it’s going to stop parsing the
  page and start over after reinterpreting the whole page using the encoding you
  specified.


Unicode Sandwich
------------------

* as soon as byte strings are brought in, decode them into unicode strings (libs like pysqlite might have done this already)
* do processing/manipulations (formatting/joining/etc..) with unicode strings
* only degrade down to an encoded byte string when writing to file/pipe/stdout
         

Python2 __unicode__ and __str__
----------------------------------

* https://stackoverflow.com/questions/1307014/python-str-versus-unicode
* see env/trac/migration/doclite.py for usage of this 

**John Millikin:**

* __str__() is the old method -- it returns bytes
* __unicode__() is the new, preferred method -- it returns characters. 

The names are a bit confusing, but in 2.x we're stuck with them for compatibility reasons. 
Generally, you should put all your string formatting in __unicode__(), and create a stub __str__() method:

::

    def __str__(self):
        return unicode(self).encode('utf-8')

In 3.0, str contains characters, so the same methods are 
named __bytes__() and __str__(). These behave as expected.

 

text out of pysqlite managed sqlite3 db
------------------------------------------

* http://pysqlite.readthedocs.io/en/latest/sqlite3.html

Encoding is fixed for the db, it cannot be changed

::

    delta:env blyth$ wtracdb-s
    -- Loading resources from /Users/blyth/.sqliterc
    SQLite version 3.17.0 2017-02-13 16:02:40
    Enter ".help" for usage hints.
    sqlite> PRAGMA encoding ; 
    encoding  
    ----------
    UTF-8     
    sqlite> 



manipulation
--------------


Single unicode format arg promotes the formatting result to unicode::

    In [4]: type(a)
    Out[4]: unicode

    In [5]: print "%s" % a
    hello

    In [6]: type("%s" % a)
    Out[6]: unicode

    In [7]: type("yo %s" % a)
    Out[7]: unicode

    In [8]: type("yo %s" % str(a))
    Out[8]: str

    In [9]: type("yo %s %s" % (a, str(a)))
    Out[9]: unicode

    In [10]: l = list(u"abc")
    Out[10]: [u'a', u'b', u'c']


    In [25]: l = list(u"abc") + map(unichr,range(0xa7,0xff+1))

    In [26]: l
    Out[26]: 
    [u'a',
     u'b',
     u'c',
     u'\xa7',
     u'\xa8',
     u'\xa9',
     u'\xaa',
     ..

    In [27]: print "".join(l)
    abc§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ

    In [28]: str("".join(l))
    ---------------------------------------------------------------------------
    UnicodeEncodeError                        Traceback (most recent call last)
    <ipython-input-28-adef8caadeff> in <module>()
    ----> 1 str("".join(l))

    UnicodeEncodeError: 'ascii' codec can't encode characters in position 3-91: ordinal not in range(128)





unicode mental model
---------------------

* https://stackoverflow.com/questions/18034272/python-str-vs-unicode-types

**Bakuriu:**

unicode is meant to handle text. Text is a sequence of code points which may be
bigger than a single byte. Text can be encoded in a specific encoding to
represent the text as raw bytes(e.g. utf-8, latin-1...).

Note that unicode is not encoded! The internal representation used by python is
an implementation detail, and you shouldn't care about it as long as it is able
to represent the code points you want.

On the contrary str in Python 2 is a plain sequence of bytes. It does not
represent text!

You can think of unicode as a general representation of some text, which can be
encoded in many different ways into a sequence of binary data represented via
str.

Note: In Python 3, unicode was renamed to str and there is a new bytes type for
a plain sequence of bytes.

Some differences that you can see between python 2 and 3::

    >>> len(u'à')  # a single code point
    1
    >>> len('à')   # by default utf-8 -> takes two bytes
    2
    >>> len(u'à'.encode('utf-8'))
    2
    >>> len(u'à'.encode('latin1'))  # in latin1 it takes one byte
    1
    >>> print u'à'.encode('utf-8')  # terminal encoding is utf-8
    à
    >>> print u'à'.encode('latin1') # it cannot understand the latin1 byte
    �


Note that using str you have a lower-level control on the single bytes of a
specific encoding representation, while using unicode you can only control at
the code-point level. For example you can do:

::

    >>> 'àèìòù'
    '\xc3\xa0\xc3\xa8\xc3\xac\xc3\xb2\xc3\xb9'
    >>> print 'àèìòù'.replace('\xa8', '')
    à�ìòù

What before was valid UTF-8, isn't anymore. Using a unicode string you cannot
operate in such a way that the resulting string isn't valid unicode text. You
can remove a code point, replace a code point with a different code point etc.
but you cannot mess with the internal representation.



handling unicode in python
-----------------------------

* http://www.utf8-chartable.de/unicode-utf8-table.pl?start=128&number=128&names=-&utf8=0x
* https://docs.python.org/2/howto/unicode.html 


::

    In [49]: for _ in range(0xa0,0xff+1):print "%4s " % hex(_), unichr(_)*100
    0xa0                                                                                                      
    0xa1  ¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡
    0xa2  ¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢¢
    0xa3  ££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££££
    0xa4  ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
    0xa5  ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
    0xa6  ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    0xa7  §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    0xa8  ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
    0xa9  ©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©©
    0xaa  ªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªªª
    0xab  ««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««
    0xac  ¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬
    0xad  ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
    0xae  ®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®®
    0xaf  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
    0xb0  °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
    0xb1  ±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±±
    0xb2  ²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²²
    0xb3  ³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³³
    0xb4  ´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´
    0xb5  µµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµ
    0xb6  ¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶
    0xb7  ····································································································
    0xb8  ¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸¸
    0xb9  ¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹
    0xba  ºººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººº
    0xbb  »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»
    0xbc  ¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼¼
    0xbd  ½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½½
    0xbe  ¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾¾
    0xbf  ¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿
    0xc0  ÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀÀ
    0xc1  ÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁÁ
    0xc2  ÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂÂ
    0xc3  ÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃÃ
    0xc4  ÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄÄ
    0xc5  ÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅÅ
    0xc6  ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ
    0xc7  ÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇÇ
    0xc8  ÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈÈ
    0xc9  ÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉÉ
    0xca  ÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊÊ
    0xcb  ËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËËË
    0xcc  ÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌÌ
    0xcd  ÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍÍ
    0xce  ÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎÎ
    0xcf  ÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏÏ
    0xd0  ÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐÐ
    0xd1  ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
    0xd2  ÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒÒ
    0xd3  ÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓÓ
    0xd4  ÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔÔ
    0xd5  ÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕÕ
    0xd6  ÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖÖ
    0xd7  ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
    0xd8  ØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØØ
    0xd9  ÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙÙ
    0xda  ÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚÚ
    0xdb  ÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛÛ
    0xdc  ÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜÜ
    0xdd  ÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝÝ
    0xde  ÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞÞ
    0xdf  ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß
    0xe0  àààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààà
    0xe1  áááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááááá
    0xe2  ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    0xe3  ãããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããããã
    0xe4  ääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääääää
    0xe5  åååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååååå
    0xe6  ææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææææ
    0xe7  çççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççççç
    0xe8  èèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèèè
    0xe9  éééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééééé
    0xea  êêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêêê
    0xeb  ëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëëë
    0xec  ìììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììììì
    0xed  íííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííííí
    0xee  îîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîîî
    0xef  ïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïïï
    0xf0  ðððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððððð
    0xf1  ññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññññ
    0xf2  òòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòòò
    0xf3  óóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóóó
    0xf4  ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô
    0xf5  õõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõõ
    0xf6  öööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööööö
    0xf7  ÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷
    0xf8  øøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøøø
    0xf9  ùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùù
    0xfa  úúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúúú
    0xfb  ûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûûû
    0xfc  üüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüüü
    0xfd  ýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýýý
    0xfe  þþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþþ
    0xff  ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ



::

    In [51]: for _ in range(0xc2a0,0xc2ff+1):print "%4s " % hex(_), unichr(_)*50
    0xc2a0  슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠슠
    0xc2a1  슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡슡
    0xc2a2  슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢슢
    0xc2a3  슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣슣
    0xc2a4  스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스스
    0xc2a5  슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥슥
    0xc2a6  슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦슦
    0xc2a7  슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧슧
    0xc2a8  슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨슨
    0xc2a9  슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩슩
    0xc2aa  슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪슪
    0xc2ab  슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫슫
    0xc2ac  슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬슬
    0xc2ad  슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭슭
    0xc2ae  슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮슮



::

    In [54]: c = unichr(0xc2a4)

    In [55]: print c
    스
     
    In [56]: ord(c)  
    Out[56]: 49828

    In [57]: ord(c) == 0xc2a4
    Out[57]: True

    In [66]: u8 = c.encode("utf-8")

    In [67]: u16 = c.encode("utf-16")

    In [68]: u32 = c.encode("utf-32")

    In [69]: print u8
    스

    In [70]: print u16
    ????

    In [71]: print u32
    ????


    In [72]: u8.decode("utf-8")
    Out[72]: u'\uc2a4'

    In [73]: u8.decode("utf-8") == c
    Out[73]: True

    In [74]: u16.decode("utf-16") == c
    Out[74]: True

    In [75]: u32.decode("utf-32") == c
    Out[75]: True

    In [76]: c
    Out[76]: u'\uc2a4'

    In [77]: hex(ord(c))
    Out[77]: '0xc2a4'


    In [78]: c2 = u"\U0000c2a4"    # \U escape needs 8 hex digits , \u needs 4 hex digits 

    In [79]: hex(ord(c2))
    Out[79]: '0xc2a4'



