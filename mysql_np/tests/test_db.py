
from env.mysql_np import DB


def test_vld_pull():
    db = DB("client")
    a = db("select * from CalibFeeSpecVld limit 10")
    print a 
    db.close()

    for field in "TIMESTART TIMEEND VERSIONDATE INSERTDATE".split():
        k = a[field].dtype.kind
        assert k == 'M' , k 

 
if __name__ == '__main__':
    test_vld_pull()



