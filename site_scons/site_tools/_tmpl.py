"""
   Fills a template , such as this __doc__ with values from a dictionary 

      2 dollars :       $$HOME    escaped \$$HOME
      1 dollar  :        $HOME    escaped  \$HOME

    Emit 1 dollar ...  $$
    Emit 2 dollar ...  $$$$

"""
import re
try:
    from string import Template   ## oops fails in python 2.3
except ImportError, e:
    from string25 import Template

class Tmpl(Template):
    pattern = re.compile(r"""
    %(delim)s(?:
      (?P<escaped>%(delim)s) |   # Escape sequence of two delimiters
      (?P<named>%(id)s)      |   # delimiter and a Python identifier
      {(?P<braced>%(expr)s)} |   # delimiter and a braced identifier
      (?P<invalid>)              # Other ill-formed delimiter exprs
    )
    """ % {"delim":r"\$", "id":r"[_a-z][_a-z0-9]*", "expr":r".*?"}, re.VERBOSE | re.IGNORECASE )


if __name__=='__main__':

    import os
    print __doc__ 
    print '-' * 60
    t = Tmpl(__doc__)
    print t.safe_substitute(os.environ)


