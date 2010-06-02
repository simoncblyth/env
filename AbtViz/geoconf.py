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
               ( 'SKIP'      , re.compile("(?P<all>logic.*|Lead.*|Worldmuon.*|Door.*|inner_log.*|Lowermuon_log_0)$") , { 'color':kOrange } ),
#
               ( 'L0'        , re.compile("(?P<all>2m_Proportional_Tube_Gas_1[6-8][0-9])$")           , { 'color':kRed    }  ),
               ( 'L1'        , re.compile("(?P<all>2m_Plastic_Scintillator_log_1[2-4][0-9])$")        , { 'color':kGreen  }  ),  
               ( 'L2'        , re.compile("(?P<all>2m_Proportional_Tube_Gas_(9[6-9]|1[0-1]\d))$")     , { 'color':kPink   }  ),
               ( 'L3'        , re.compile("(?P<all>1.5m_Plastic_Scintillator_log_[6-8]\d)$")          , { 'color':kYellow }  ),     
               ( 'L4'        , re.compile("(?P<all>2m_Proportional_Tube_Gas_(3[2-9]|[4-5]\d))$")      , { 'color':kBlue   }  ),
               ( 'L5'        , re.compile("(?P<all>1m_Plastic_Scintillator_log_(\d|1[0-4]))$")        , { 'color':kTeal   }  ),
               ( 'L6'        , re.compile("(?P<all>1m_Plastic_Scintillator_log_(1[5-9]|2\d))$")       , { 'color':kSpring }  ),
#
               ( 'FILM'      , re.compile("(?P<all>(?P<pos>top|bot)_film_log_0)$")                   , { 'color':kViolet } ),                   
               ( 'WRLD'      , re.compile("(?P<all>World_1|Worldneutron_log_0|expHall_log_0)$")      , { 'color':kGray   } ),
               ( 'SST'       , re.compile("(?P<all>steeltank_log_0)$")                               , { 'color':kTeal   } ),
               ( 'TOPS'      , re.compile("(?P<all>outer_log_0|acrylictank_log_0)$")                 , { 'color':kSpring } ),
               ( 'PS1'       , re.compile("(?P<all>1m_Plastic_Scintillator_log_(?P<id>\d*))$")       , { 'color':kRed    } ),
               ( 'PS15'      , re.compile("(?P<all>1.5m_Plastic_Scintillator_log_(?P<id>\d*))$")     , { 'color':kGreen  } ),
               ( 'PT2'       , re.compile("(?P<all>2m_Proportional_Tube_Gas_(?P<id>\d*))$")          , { 'color':kPink   } ),
               ( 'PT2F'      , re.compile("(?P<all>2m_Proportional_Tube_(?P<id>\d*))$")              , { 'color':kYellow } ), 
               ( 'PS2'       , re.compile("(?P<all>2m_Plastic_Scintillator_log_(\d*))$")             , { 'color':kBlue   } ),   ## same name for leaf and its folder 
#   
               ( 'catchall'  , re.compile("(?P<all>.*$)")                                            , { 'color':kAzure  } ),
            )



"""

    for v in g.geom.geo.rmatch( "2m_Proportional_Tube_Gas_1[6-8][0-9]$"):v.SelectElement(kTRUE)

"""



