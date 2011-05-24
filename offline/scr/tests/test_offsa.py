from off import OffTableName as OTN, instances
from offsa import OFF

off = None
def setup():
    global off
    off = OFF("recovered_offline_db")

def teardown():
    pass

def test_dynamic_pull():
    print "lookup dynamic classes from OTN coordinates "
    for qty in "PmtHv AdTemp".split():
        kls = off.kls(OTN("Dcs",qty,"Vld"))
        print kls, kls.__name__

def test_filter():
    for otn in instances(): 
        kls = off.kls( otn )
        assert str(kls.xtn) == str(otn), (kls.xtn, otn )    ## kls knows where it came from
        assert kls.db  == off 

        print "first %r " % kls.first()
        print "last  %r " % kls.last()
        print "count %r " % kls.count()

        print "%" * 20, otn, "%" * 20

        #qa = off.qa(kls)
        #qd = off.qd(kls)
        qa = kls.qa()    ## using classmethod shortcut
        qd = kls.qd()

        la = qa.first()   
        print "first in SEQNO ascending order, ie 1st SEQNO", la, la.SEQNO
        ld = qd.first()   
        print "first in SEQNO descending order, ie last SEQNO", ld, ld.SEQNO
        na = qa.count()
        nd = qd.count()
        assert na == nd
        print "qa count %d qd count %d " % ( na, nd )

        if otn.flv == "Vld":
            cut = ld.SEQNO - 3
            print "SEQNO after %s " % cut 
            for i in qa.filter(kls.SEQNO > cut ).all():
                print i

def test_manual_join():
    print "manual join"
    j = off._join("DcsPmtHv","DcsPmtHvVld", "SEQNO", "V_")
    print "j", j

def test_auto_join_and_map():
    print "auto join and map" 
    otn = OTN("Dcs","AdTemp","Pay:Vld:SEQNO:V_") 
    kls = off.kls(otn)
    print kls

def test_last():
    last = kls.last()
    print "last", last, dir(last)
    

if __name__ == '__main__':
    setup()

    test_dynamic_pull()
    test_filter()
    test_manual_join()
    test_auto_join_and_map()

    teardown()
