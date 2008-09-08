"""
   python cleaner.py < cleaner.py


"""

class Cleaner:
    def __init__(self):
        """
          Get rid of those pesky terminal escape color codes ... 
          as they invalidate xml CDATA sections that attempt to contain them 
    
            \033[22;30m - black
            \033[22;31m - red
            \033[22;32m - green
            \033[22;33m - brown
            \033[22;34m - blue
            \033[22;35m - magenta
            \033[22;36m - cyan
            \033[22;37m - gray
            \033[01;30m - dark gray
            \033[01;31m - light red
            \033[01;32m - light green
            \033[01;33m - yellow
            \033[01;34m - light blue
            \033[01;35m - light magenta
            \033[01;36m - light cyan
            \033[01;37m - white

        """
        import re
        self.rm= re.compile("\033\[[0-9;]+m") 

    def __call__(self, data ):
        """  replaces all matches of the regexp in data with an empty string  """
        return self.rm.sub("", data)   


if __name__=='__main__':
    import sys
    cl = Cleaner()
    for line in sys.stdin.read().splitlines(True):
        sys.stdout.write(cl(line))
