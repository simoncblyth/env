
MySQL_NumPy
============

State of dependencies


#. MySQL-python now has a 1.2.4b5 (currently using 1.2.3)

  * http://mysql-python.blogspot.tw/  bugfix release only

#. can it work with NumPy 1.6.2 (originally used development NumPy 2.0.0, and pushed some changes upstream) 

  

::

        commit bcbd02bbb00d29ce4004665bcf34158913a6ef2c
        Author: Simon Blyth <simon.c.blyth@gmail.com>
        Date:   Wed Dec 8 19:47:36 2010 +0800

            apply the MySQLdb-1.2.3-resultiter.patch ... adding numpy 2.0.0 dependency (for datetime support)



How is datetime support in current numpy 1.6.2

   * from macports installed source, there is reason to be hopeful and try it out 



macports numpy 1.6.2, a dependency of matplotlib
-----------------------------------------------------

::

        py26-matplotlib @1.1.1, Revision 2 (python, graphics, math)


::

        simon:MySQLdb blyth$ ipython-2.6
        Python 2.6.8 (unknown, Oct 24 2012, 14:09:39) 
        Type "copyright", "credits" or "license" for more information.

        IPython 0.12 -- An enhanced Interactive Python.
        ?         -> Introduction and overview of IPython's features.
        %quickref -> Quick reference.
        help      -> Python's own help system.
        object?   -> Details about 'object', use 'object??' for extra details.

        In [1]: import numpy as np

        In [2]: np.version.version
        Out[2]: '1.6.2'

        In [3]: np.get_include()
        Out[3]: '/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include'



try to build my mysql_numpy against this
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    cd
    git clone git://github.com/scb-/mysql_numpy.git


Make sure to run `setup.py` with the appropriate python::

    cd ~/mysql_numpy/MySQLdb

    simon:MySQLdb blyth$ python2.6 -c "import numpy as np ; print np.version.version, np.get_include() "
    1.6.2 /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include

Build and install `_mysql.so`::

        simon:MySQLdb blyth$ python2.6 setup.py build 
        ...
        simon:MySQLdb blyth$ sudo python2.6 setup.py install
        ...
        Installed /opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/MySQL_python-1.2.3-py2.6-macosx-10.5-ppc.egg
        Processing dependencies for MySQL-python==1.2.3
        Finished processing dependencies for MySQL-python==1.2.3


against nuwapy on belle7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

        [blyth@belle7 ~]$ git clone git://github.com/scb-/mysql_numpy.git
        ...
        [blyth@belle7 DybDbi]$ cd ~/mysql_numpy/MySQLdb
        [blyth@belle7 MySQLdb]$ which python
        /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python

        [blyth@belle7 MySQLdb]$ python setup.py install





interactive test from G
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

        simon:MySQLdb blyth$ ipython-2.6
        Python 2.6.8 (unknown, Oct 24 2012, 14:09:39) 
        Type "copyright", "credits" or "license" for more information.

        IPython 0.12 -- An enhanced Interactive Python.
        ?         -> Introduction and overview of IPython's features.
        %quickref -> Quick reference.
        help      -> Python's own help system.
        object?   -> Details about 'object', use 'object??' for extra details.

        In [1]: import _mysql

        In [8]: conn = _mysql.connect( read_default_group="client" )

        In [9]: conn.query("select * from DcsAdWpHvVld")

        In [10]: r = conn.store_result()

        In [11]: a = r.fetch_nparray(verbose=0)

        In [12]: print repr(a)
        array([ (1, datetime.datetime(2012, 7, 1, 3, 0), datetime.datetime(2012, 7, 2, 3, 0), 1, 1, 1, 0, -1, datetime.datetime(2012, 7, 1, 3, 0), datetime.datetime(2012, 10, 16, 20, 51, 44)),
               (2, datetime.datetime(2012, 7, 1, 3, 0), datetime.datetime(2012, 7, 2, 3, 0), 1, 1, 2, 0, -1, datetime.datetime(2012, 7, 1, 3, 0), datetime.datetime(2012, 10, 16, 20, 51, 44)),
               (3, datetime.datetime(2012, 7, 1, 3, 0), datetime.datetime(2012, 7, 2, 3, 0), 2, 1, 1, 0, -1, datetime.datetime(2012, 7, 1, 3, 0), datetime.datetime(2012, 10, 16, 20, 51, 44)),
               ...,
               (1336, datetime.datetime(2012, 10, 13, 15, 15, 17), datetime.datetime(2012, 10, 15, 15, 18, 15), 2, 1, 6, 0, -1, datetime.datetime(2012, 10, 13, 15, 15, 17), datetime.datetime(2012, 10, 24, 6, 1, 53)),
               (1337, datetime.datetime(2012, 9, 17, 1, 35, 6), datetime.datetime(2012, 9, 18, 8, 36, 23), 1, 1, 2, 0, -1, datetime.datetime(2012, 9, 17, 1, 35, 6), datetime.datetime(2012, 10, 24, 8, 21, 25)),
               (1338, datetime.datetime(2012, 9, 18, 8, 36, 23), datetime.datetime(2012, 9, 22, 2, 4, 46), 1, 1, 2, 0, -1, datetime.datetime(2012, 9, 18, 8, 36, 23), datetime.datetime(2012, 10, 24, 8, 21, 56))], 
              dtype=[('SEQNO', '>i4'), ('TIMESTART', ('>M8[s]', {})), ('TIMEEND', ('>M8[s]', {})), ('SITEMASK', '>i2'), ('SIMMASK', '>i2'), ('SUBSITE', '>i4'), ('TASK', '>i4'), ('AGGREGATENO', '>i4'), ('VERSIONDATE', ('>M8[s]', {})), ('INSERTDATE', ('>M8[s]', {}))])

        In [13]: 





fast test
~~~~~~~~~~~~

::

        #!/usr/bin/env python

        import _mysql

        icol = "SITEMASK SIMMASK SUBSITE TASK AGGREGATENO".split() 
        tcol = "TIMESTART TIMEEND VERSIONDATE INSERTDATE".split()
        coerce = dict(map(lambda _:(_,"M8[s]"),tcol))
        ut_ = lambda _:"UNIX_TIMESTAMP(%(_)s) as %(_)s" % locals() 

        #cols = ",".join( icol + map(ut_,tcol ) )
        cols = ",".join( icol )

        vld_ = lambda t,c:"select %(c)s from %(t)sVld" % locals() 

        if __name__ == '__main__':
            conn = _mysql.connect(read_default_group="client")
            sql = vld_("DcsAdWpHv",cols)
            print sql
            conn.query(sql)
            r = conn.store_result()
            a = r.fetch_nparrayfast(verbose=1,coerce=coerce)
            print repr(a)
            conn.close()



With times included, get SystemError::

        simon:MySQLdb blyth$ python2.6 fast.py 
        {'TIMESTART': 'M8[s]', 'INSERTDATE': 'M8[s]', 'VERSIONDATE': 'M8[s]', 'TIMEEND': 'M8[s]'}
        select SITEMASK,SIMMASK,SUBSITE,TASK,AGGREGATENO,UNIX_TIMESTAMP(TIMESTART) as TIMESTART,UNIX_TIMESTAMP(TIMEEND) as TIMEEND,UNIX_TIMESTAMP(VERSIONDATE) as VERSIONDATE,UNIX_TIMESTAMP(INSERTDATE) as INSERTDATE from DcsAdWpHvVld limit 5 
        _fetch_nparrayfast
        coerce fields named TIMESTART with numpy dtype code ... 'M8[s]'
        coerce fields named TIMEEND with numpy dtype code ... 'M8[s]'
        coerce fields named VERSIONDATE with numpy dtype code ... 'M8[s]'
        coerce fields named INSERTDATE with numpy dtype code ... 'M8[s]'
        descr
        [('SITEMASK', 'h2'), ('SIMMASK', 'h2'), ('SUBSITE', 'i4'), ('TASK', 'i4'), ('AGGREGATENO', 'i4'), ('TIMESTART', 'M8[s]'), ('TIMEEND', 'M8[s]'), ('VERSIONDATE', 'M8[s]'), ('INSERTDATE', 'M8[s]')]
         npdescr using PyArray_DescrConverter 
         dtype
         dtype([('SITEMASK', '>i2'), ('SIMMASK', '>i2'), ('SUBSITE', '>i4'), ('TASK', '>i4'), ('AGGREGATENO', '>i4'), ('TIMESTART', ('>M8[s]', {})), ('TIMEEND', ('>M8[s]', {})), ('VERSIONDATE', ('>M8[s]', {})), ('INSERTDATE', ('>M8[s]', {}))])
         _fetch_nparrayfast fields 9 elsize 48 nele 5 size 240 
          i 0  row[i] 1 offset 0 fmt %hd type 3 NPY_SHORT rc 1 inbuffer 1
          i 1  row[i] 1 offset 2 fmt %hd type 3 NPY_SHORT rc 1 inbuffer 1
          i 2  row[i] 1 offset 4 fmt %ld type 7 NPY_LONG rc 1 inbuffer 1
          i 3  row[i] 0 offset 8 fmt %ld type 7 NPY_LONG rc 1 inbuffer 0
          i 4  row[i] -1 offset 12 fmt %ld type 7 NPY_LONG rc 1 inbuffer -1
          i 5  row[i] 1341136800 offset 16 fmt NPY_STRING_FMT type 21 NPY_STRING rc 0 inbuffer NPY_STRING_FMT
          field 5 scan failure 
          nparray_fast error...  
          Traceback (most recent call last):
          File "fast.py", line 24, in <module>
          a = r.fetch_nparrayfast(verbose=3,coerce=coerce)
          SystemError: error return without exception set
          simon:MySQLdb blyth$ 

Type mess up the time integer is getting handled as string::

        simon:MySQLdb blyth$ echo "select  UNIX_TIMESTAMP(TIMESTART) as TIMESTART from DcsAdWpHvVld limit 1" | mysql -t
        +------------+
        | TIMESTART  |
        +------------+
        | 1341136800 |
        +------------+

Maybe feeling

  * http://docs.scipy.org/doc/numpy-dev/reference/arrays.datetime.html#differences-between-1-6-and-1-7-datetimes




Without times getr Bus error::

        simon:MySQLdb blyth$ vi fast.py 
        simon:MySQLdb blyth$ python2.6 fast.py 
        select SITEMASK,SIMMASK,SUBSITE,TASK,AGGREGATENO from DcsAdWpHvVld
        npdescr using PyArray_DescrConverter 
        _fetch_nparrayfast fields 5 elsize 16 nele 1338 size 21408 
        Bus error







