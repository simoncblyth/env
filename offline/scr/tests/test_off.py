
from off import OffTableName as OTN

def instances():
    """Selection of OffTableName instances"""
    cls = OTN
    otns = []
    for pfx in cls.pfxList:
        for qty in cls.qtyList:
            for flv in cls.flvList:
                otns.append( cls(pfx,qty,flv) )
    return otns
   
def test_tables():
    tabs = map(str, instances() )

def test_classes():
    clss = map(lambda _:_.kln, instances() )

def test_otn():
    cls = OTN
    otn = cls("Dcs", "PmtHv", "Pay" )
    print otn, repr(otn)
    for otn in instances():
        print otn, repr(otn)

def test_otn_join():
    otn = OTN("Dcs", "PmtHv", "Pay:Vld:SEQNO:V_" )
    print otn, repr(otn)
    assert otn.isjoin
    for jtn in otn.jbits():
       print jtn,repr(jtn)

if __name__ == '__main__':


    test_tables()
    test_classes()
    test_otn()
    test_otn_join()
