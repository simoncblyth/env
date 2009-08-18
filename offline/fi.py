"""
       somewhat of a perl -ism 
          http://effbot.org/librarybook/fileinput.htm


       inplace editing of files 


"""
        

class Edit:
    def __call__(self,path):
        import fileinput as fi
        for line in fi.input(path,inplace=0):self.filter( line,fi.lineno() )
        return self

class Odd(Edit):
    def filter(self,line,n):
        if n in [1,3,5]:print "%s %s" % ( n , line ),

def make(path):
    f = file(path,"w")
    for n in range(10,20):f.write("%s\n"%str(n))


if __name__=='__main__':
    path = "odd.txt"
    make(path)
    Odd()(path)

 
