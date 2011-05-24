
from datetime import datetime
from dcs import DcsTableName as DTN
from dcssa import DCS

dcs = None

def setup():
    global dcs
    dcs = DCS("dcs")

def teardown():
    for t in dcs.meta.tables:
        print t 

def test_filter():
    cut = datetime( 2011,5,19, 9 )
    dtns = []
    for site in "DBNS".split():
        for det in "AD1 AD2".split():
            for qty in "HV HV_Pw".split():
                dtns.append( DTN(site, det, qty) )

    for dtn in dtns: 
        print "%" * 20, dtn, "%" * 20

        kls = dcs.kls(dtn)
        assert kls.db  == dcs
        assert kls.xtn == dtn

        print "first %r " % kls.first()
        print "last  %r " % kls.last()
        print "count %r " % kls.count()

        qa = dcs.qa(kls)
        qd = dcs.qd(kls)

        print "date_time ascending %s" % kls.__name__
        for i in qa.all():
            print i 
        print "date_time descending %s" % kls.__name__
        for i in qd.all():
            print i 
        print "before %s " % cut    ## hmmm a qbefore / qafter would avoid having to spill the kls
        for i in qa.filter(kls.date_time < cut ).all():
            print i
        print "after %s " % cut 
        for i in qa.filter(kls.date_time > cut ).all():
            print i

def test_dynamic_pull():
    print "dynamic classes from DTN coordinates "
    for det in "AD1 AD2".split():
        for qty in "HV HV_Pw".split(): 
            dtn = DTN("DBNS", det , qty )
            kls = dcs.kls(dtn)
            assert str(kls.xtn) == str(dtn), "dtn %s xtn %s" % (dtn, kls.xtn)
            assert kls.db  == dcs
            print kls, kls.__name__

def test_autojoin_last():
    print "autojoin"
    dtn = DTN("DBNS","AD1","HV:HV_Pw:id:P_")
    kls = dcs.kls(dtn)
    last = kls.last()
    for k,v in sorted(last.asdict.items()):
        print k,v
 

if __name__ == '__main__':

    setup()

    test_filter()
    test_dynamic_pull() 
    test_autojoin_last()

    teardown()


