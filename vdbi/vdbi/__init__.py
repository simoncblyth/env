

from vdbi.app.ip_vdbi import ip_vdbi
from vdbi.app         import serve_app

PAY_COLUMNS = {
   'SEQNO':'SEQ',
   'ROW_COUNTER':'ROW',
   }

VLD_COLUMNS = {
   'SEQNO':'VSEQ',
   'TIMESTART':'VSTART',
   'TIMEEND':'VEND',
   'SITEMASK':'VSITE',
   'SIMMASK':'VSIM',
   'SUBSITE':'VSUB',
   'TASK':'VTASK',
   'AGGREGATENO':'VAGNO',
   'VERSIONDATE':'VVERS',
   'INSERTDATE':'VINSERT',
   }
   
   
VLD_TIMEATTS = map( lambda x:VLD_COLUMNS[x] ,  ('INSERTDATE','TIMEEND','TIMESTART','VERSIONDATE',) )


## cannot use app here ??
def is_vld_resource(resource):
    name = getattr( resource , '__name__' , None)
    return name and name.endswith('Vld')

def is_dbi_resource(resource):
    name = getattr( resource , '__name__' , None)
    return name and name.endswith('Dbi')


def get_default_x(resource):
    if is_vld_resource(resource):
        return PAY_COLUMNS['SEQNO']   ## *Vld have SEQ attr not VSEQ ... due to some special FK treatment
    elif is_dbi_resource(resource):
        return VLD_COLUMNS['TIMESTART'] 
    else:
        return PAY_COLUMNS['SEQNO']
    
def get_default_y(resource):
    if is_vld_resource(resource):
        return VLD_COLUMNS['TIMESTART'] 
    elif is_dbi_resource(resource):
        return PAY_COLUMNS['ROW_COUNTER']
    else:
        return PAY_COLUMNS['ROW_COUNTER']

def dbi_default_plot( resource ):
    return [{ 'x':get_default_x(resource), 'y':get_default_y(resource)}]  



CTX_COLUMNS = ('SITEMASK','SUBSITE','SIMMASK',)
CTX_KEYS    = ('Site', 'DetectorId','SimFlag',)

JOIN_POSTFIX = 'Dbi'
VLD_POSTFIX = 'Vld'

JOIN_POSTFIX_ = JOIN_POSTFIX.lower()
VLD_POSTFIX_ = VLD_POSTFIX.lower()

def is_vld_attr( attr ):
    return attr in VLD_COLUMNS.values()

def is_vld_column( coln ):
    return coln in VLD_COLUMNS.keys()
