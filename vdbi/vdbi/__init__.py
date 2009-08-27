
from vdbi.app.ip_vdbi import ip_vdbi
from vdbi.app         import serve_app

from IPython.Debugger import Tracer
debug_here = Tracer()

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

JOIN_POSTFIX = 'Dbi'
VLD_POSTFIX = 'Vld'

JOIN_POSTFIX_ = JOIN_POSTFIX.lower()
VLD_POSTFIX_ = VLD_POSTFIX.lower()

def is_vld_attr( attr ):
    return attr in VLD_COLUMNS.values()

def is_vld_column( coln ):
    return coln in VLD_COLUMNS.keys()
