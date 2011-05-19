
class OffTableName(object):
    """
    Supply Offline DB table names, given some ctor context arguments (tbd)
    and attribute qty

    This class is intended as insulation against table name changes.

    Old scraper code features "DcsAdPmtHv" which does not appear 
    in offline_db 
    """ 
    names = dict(temp="DcsAdTemp", hv="DcsPmtHv")
    def __getattr__(self, name):
        if name in self.names:
            return self.names[name] 
        else:
            raise AttributeError

    def dbi_pairs(self, names ):
        ret = []
        for name in names:
            tn = getattr(self, name) 
            ret.append( tn )
            ret.append( tn + "Vld" )
        return ret

class Base(object):
    """
    Base for mapped classes that have a SEQNO attribute
    """
    def __repr__(self):
        return "%s %s " % ( self.__class__.__name__, self.SEQNO )

class Vld(Base):
    def __repr__(self):
        return "%s %s %s %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.TIMESTART, self.TIMEEND, self.VERSIONDATE, self.INSERTDATE )


if __name__ == '__main__':
    pass

    from sa import SA
    otn = OffTableName()
    off = SA("recovered_offline_db", tables=otn.dbi_pairs(["hv"])  )




