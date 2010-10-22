import re
import os
from ROOT import kWhite, kBlack, kGray, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan, kOrange, kSpring, kTeal, kAzure, kViolet, kPink
from geoedit import Matcher

class GeoConf:
    rawpath = "$ENV_HOME/aberdeen/root/Aberdeen_World.root"
    xname = "extract"
    xpath = os.path.basename(rawpath).replace(".","_%s." % xname )
    
class VolMatcher(Matcher):
    """
         The first pattern that matches a volumes name is used ... so the order or the 
         patterns is vital 
 
         Colors are only applied at the prepare_geom.py stage ... so color changes will
         not be reflected until that is rerun, creating a new *_extract.root

    """
   
    patns = (
               ( 'SKIP'      , re.compile("(?P<all>logic.*|Lead.*|Worldmuon.*|Door.*|inner_log.*|Lowermuon_log_0|reflectivelayer_log_0)$") , { 'color':kOrange } ),
#
               
               	( 'L1'        , re.compile("(?P<all>1m_Plastic_Scintillator_log_[0-9])$")    	, { 'color':kSpring  }  ),  
		( 'L0'        , re.compile("(?P<all>1.5m_Plastic_Scintillator_log_(1[6-9]|2[0-5]))$")      	, { 'color':kBlue   }  ),
 		( 'L2'        , re.compile("(?P<all>1m_Plastic_Scintillator_log_(3[2-9]|4[0-1]))$")          	, { 'color':kYellow }  ),  
               	( 'L3'        , re.compile("(?P<all>1.5m_Plastic_Scintillator_log_(4[8-9]|5[0-7]))$")     	, { 'color':kPink   }  ),
               	( 'L6'        , re.compile("(?P<all>1m_Plastic_Scintillator_log_(6[4-9]|7[0-3]))$")    , { 'color':kCyan   }  ),
               	#( 'L5'        , re.compile("(?P<all>1m_Plastic_Scintillator_log_(14[3-9]|15[0-7]))$")      , { 'color':kSpring }  ),
		( 'L5'        , re.compile("(?P<all>2m_Plastic_Scintillator_log_(8[0-9]|9[0-3]))$")           	, { 'color':kRed    }  ),
#         
               ( 'FILM'      , re.compile("(?P<all>(?P<pos>top|bot)_film_log_([0-1]))$")                   , { 'color':kViolet } ),                   
               ( 'WRLD'      , re.compile("(?P<all>World_1|Worldneutron_log_0|expHall_log_0)$")      , { 'color':kGray   } ),
               ( 'SST'       , re.compile("(?P<all>steeltank_log_0)$")                               , { 'color':kTeal   } ),
               ( 'TOPS'      , re.compile("(?P<all>outer_log_0|acrylictank_log_0)$")                 , { 'color':kSpring } ),
               ( 'PS1'       , re.compile("(?P<all>1m_Plastic_Scintillator_log_(?P<id>\d*))$")       , { 'color':kWhite    } ),
               ( 'PS15'      , re.compile("(?P<all>1.5m_Plastic_Scintillator_log_(?P<id>\d*))$")     , { 'color':kGreen  } ),
               ( 'PT2'       , re.compile("(?P<all>2m_Proportional_Tube_Gas_(?P<id>\d*))$")          , { 'color':kBlack   } ),
               ( 'PT2F'      , re.compile("(?P<all>2m_Proportional_Tube_(?P<id>\d*))$")              , { 'color':kPink } ), 
               ( 'PS2'       , re.compile("(?P<all>2m_Plastic_Scintillator_log_(\d*))$")             , { 'color':kViolet   } ),   ## same name for leaf and its folder 
#   
               ( 'catchall'  , re.compile("(?P<all>.*$)")                                            , { 'color':kAzure  } ),
            )



"""

    for v in g.geom.geo.rmatch( "2m_Proportional_Tube_Gas_1[6-8][0-9]$"):v.SelectElement(kTRUE)

"""



