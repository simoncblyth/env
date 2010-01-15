
class V(list):
    tmpl = (
{'Extra': '', 'Default': '0', 'Field': 'SEQNO', 'Key': '', 'Null': 'NO', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'TIMESTART', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'TIMEEND', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}, 
{'Extra': '', 'Default': None, 'Field': 'SITEMASK', 'Key': '', 'Null': 'YES', 'Type': 'tinyint(4)'}, 
{'Extra': '', 'Default': None, 'Field': 'SIMMASK', 'Key': '', 'Null': 'YES', 'Type': 'tinyint(4)'}, 
{'Extra': '', 'Default': None, 'Field': 'SUBSITE', 'Key': '', 'Null': 'YES', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': None, 'Field': 'TASK', 'Key': '', 'Null': 'YES', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': None, 'Field': 'AGGREGATENO', 'Key': '', 'Null': 'YES', 'Type': 'int(11)'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'VERSIONDATE', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}, 
{'Extra': '', 'Default': '0000-00-00 00:00:00', 'Field': 'INSERTDATE', 'Key': '', 'Null': 'NO', 'Type': 'datetime'}
          )

    def __init__(self):
        for f in self.__class__.tmpl:
            self.append( f )

    def __getitem__(self, k ):
        """
              Provide flexible access to table description ...
                   v[0] v[-1]      dict for 1st/last field
                   v['SEQNO']      dict for the SEQNO field
                   v['Type']       list of types in field order

        """ 
        if type(k) == int:
            return list.__getitem__(self, k)
        for f in self:
            if f['Field'] == k:return f
        for c in self[0].keys():
            if c == k:
                return [f[k] for f in self]
        return None     



if __name__=='__main__':
    v = V()    
    print v[0].keys()   

    for k in v[0].keys():
        print v[k]

    print v[0]
    print v[-1]
    print v['SEQNO']
    print v['SIMMASK']


