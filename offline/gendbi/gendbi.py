"""
  Fills templates with the context parsed from csv format spec files 
  
  meta          ,  class         ,  table       ,  CanL2Cache 
  1             ,  SimPmtSpec    ,  SimPmtSpec  ,  kTRUE
; 
  name           , codetype                 , dbtype       , description                          , code2db 
  pmtId          , DayaBay::DetectorSensor  , int(11)      , PMT sensor ID                        , .sensorId()
  describ        , std::string              , varchar(27)  , String of decribing PMT position     ,
  gain           , double                   , float        , Relative gain for pmt with mean = 1  ,


  Use by piping in the spec with a single argument naming the template 

     cat SimPmtSpec.spec | python gendbi.py SubDbiTableRow.h   > SimPmtSpec.h
     cat SimPmtSpec.spec | python gendbi.py SubDbiTableRow.sql > SimPmtSpec.sql

"""
from parse import Tab
from tmpl import filltmpl

def fill( spec, template ):
	t = Tab.parse_csv( spec )
	return filltmpl( template , t , cls=t.meta.get('class','ErrorNoClass'))

if __name__=='__main__':
	import sys
	print len(sys.argv) > 1 and fill( sys.stdin , sys.argv[1] ) or __doc__

