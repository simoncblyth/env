"""
    Fixups for the models streamed from 
        dj-manage inspectdb

    this rearranges the order of the classes and 

"""

class Matcher(dict):
    def __init__(self):
        import re
        self['_cls'] = re.compile("^class (?P<cls>\S*)\(models.Model\):")        
        self['_fk']  = re.compile("^    seqno = models.IntegerField\(null=True, db_column=\'(?P<fk>\S*)\'.*")
        self['_pk']  = re.compile("^    row_counter = models.IntegerField\(null=True, db_column=\'(?P<pk>\S*)\', blank=True\).*")

    def __call__(self, name , line):
        m = self[name].match(line)
        if m:self.update(m.groupdict())
        return self


def FK(related, fk ):
    tmpl = """    seqno = models.ForeignKey(%(related)s, db_column='%(fk)s', verbose_name='vld' ) # changed by fix.py \n"""
    return tmpl % locals()    

def PK( pk ):
    tmpl = """    row_counter = models.IntegerField(primary_key=True , db_column='%(pk)s') # changed by fix.py \n"""
    return tmpl % locals()    


class Fix(dict):
    def __init__(self): 
        self.m = Matcher()
    
    def __call__(self,lines):
        for n,line in enumerate(lines):self.filter( line, n )
        return self

    def filter(self,line,n):
        cls = self.m('_cls',line).get('cls',None)
        if cls and ( cls.startswith('Auth') or cls.startswith('Django') ):
            pass
        else:
            if not(cls):cls='_HEAD'
            if not self.get(cls,None):self[cls] = []
            self.cls = cls
            line = self.subs(line)
            self[cls].append( line)
            #print line,

    def subs(self,line):
        fk = self.m('_fk', line).get('fk',None)
        if fk:
            self.m['fk'] = None
            return FK( "%svld" % self.cls , fk )  

        pk = self.m('_pk', line).get('pk',None)
        if pk:
            self.m['pk'] = None
            return PK( pk )  
 
        return line

    def __repr__(self):
        """
            vld must come before payload, as it is referred to from the payload FK
        """
        lines = self['_HEAD']
        names = self.keys()
        vld = [n for n in self.keys() if n.endswith('vld')]
        pay = [n[0:-3] for n in vld if self.get(n[0:-3],None)] 
        
        for v,p in zip(vld,pay):
            lines.extend(self[v] + self[p])
        return "".join(lines)



if __name__=='__main__':
    import sys
    print Fix()(sys.stdin.readlines())

