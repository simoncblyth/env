
from dcs import DcsTableName as DTN
 

def instances():
    """Selection of DcsTableName instances"""
    dtns = []
    for site in "DBNS".split():
        for det in "AD1 AD2".split():
            for qty in "HV HV_Pw".split():
                dtns.append( DTN(site, det, qty) )
    return dtns


def test_tables():    
    return map(str, instances() )
def test_classes():
    return map(lambda _:_.kln, instances() )

def test_dtn():            
    cls = DTN
    dtn = cls("DBNS", "AD1", "HV" )
    print dtn, repr(dtn)
    for dtn in instances():
        print dtn, repr(dtn)

            
if __name__ == '__main__':
    test_tables()
    test_classes()
    test_dtn()

