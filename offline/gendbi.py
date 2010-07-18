import csv
from string import strip

class Qwn(dict):
    pass

class Tab(list):
    @classmethod
    def parse_csv(cls, src):
        """
            parse csv spec into Tab object comprising 
                * .meta dict of properties for table level data
                * list of dicts for row level data 

            lines with ";" reset fieldnames allowing table 
            and row level data to be handled together

            whitespace at front and back of keys and values is stripped              
        """ 
        t = Tab() 
        rdr = csv.DictReader( src )
        for r in rdr:
            ff = rdr.fieldnames and r[rdr.fieldnames[0]] or None
            if ff[0] == ";":
                rdr.fieldnames = None
                continue
            d = dict(map(lambda _:map(strip,_), r.items() ))
            if d.has_key('meta'):
                t.meta = d
            else:
                t.append(Qwn(d))
        return t

    def __repr__(self):
        return "\n".join([repr(self.meta),list.__repr__(self)])

if __name__=='__main__':
    import sys
    t = Tab.parse_csv(sys.stdin)
    print t
