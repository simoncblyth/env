
from vdbi.app.ip_vdbi import ip_vdbi
from vdbi.app         import serve_app

debug_here = lambda : None
try:
    import IPython
    debug_here = IPython.Debugger.Tracer()
except ValueError: 
    None


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
