
class V(list):
    """
        Override list behaviour to provide flexible access 
        to table description ...
            v[0] v[-1]      dict for 1st/last field
            v['SEQNO']      dict for the SEQNO field
            v['Type']       list of types in field order
    """ 

    fields =  ['SEQNO', 'TIMESTART', 'TIMEEND', 'SITEMASK', 'SIMMASK', 'SUBSITE', 'TASK', 'AGGREGATENO', 'VERSIONDATE', 'INSERTDATE']
    #enumfields  = { 'Site':'SITEMASK' , 'DetectorId':'SUBSITE' , 'SimFlag':'SIMMASK'  }
    enumfields  = { 'SITEMASK':'Site' , 'SUBSITE':'DetectorId' , 'SIMMASK':'SimFlag' }
    datefields  = ( 'TIMESTART', 'TIMEEND', 'VERSIONDATE', 'INSERTDATE' )

    @classmethod
    def all_asserts(cls):return [m for m in dir(cls) if m.startswith('assert_')]

    def __getitem__(self, k ):
        if type(k) == int:
            return list.__getitem__(self, k)
        for f in self:
            if f['Field'] == k:return f
        for c in self[0].keys():
            if c == k:
                return [f[k] for f in self]
        return None     


    def assert_fields(self):    assert self['Field'] == V.fields , "Invalid Fields"
    def assert_pk_pri(self):    assert self['SEQNO']['Key'] == 'PRI' , "SEQNO must be primary key"
    def assert_pk_nonnull(self):assert self['SEQNO']['Null'] == 'NO' , "SEQNO cannot be Null"



if __name__=='__main__':

    print V.all_asserts()

    v = V((
{'Extra': '', 'Default': '0', 'Field': 'SEQNO', 'Key': 'PRI', 'Null': 'NO', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'TIMESTART', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'TIMEEND', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}, 
{'Extra': '', 'Default': None, 'Field': 'SITEMASK', 'Key': '', 'Null': 'YES', 'Type': 'tinyint(4)'}, 
{'Extra': '', 'Default': None, 'Field': 'SIMMASK', 'Key': '', 'Null': 'YES', 'Type': 'tinyint(4)'}, 
{'Extra': '', 'Default': None, 'Field': 'SUBSITE', 'Key': '', 'Null': 'YES', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': None, 'Field': 'TASK', 'Key': '', 'Null': 'YES', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': None, 'Field': 'AGGREGATENO', 'Key': '', 'Null': 'YES', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'VERSIONDATE', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'INSERTDATE', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}
          ))

    print v[0].keys()   

    for k in v[0].keys():
        print v[k]

    print v[0]
    print v[-1]
    print v['SEQNO']
    print v['SIMMASK']

    v.assert_pk()
    v.assert_fields()


