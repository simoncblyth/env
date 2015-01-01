#!/bin/env python
"""
flags.py
===========

C++ code generation for bitfield flag handling.
expects path to json file argument of format::

    {
       "G4DAEChroma": [
                        ["FLAG_G4SCINTILLATION_ADD_SECONDARY" ,"1 << 0"], 
                        ["FLAG_G4SCINTILLATION_KILL_SECONDARY","1 << 1"], 
                        ["FLAG_G4SCINTILLATION_COLLECT_STEP",  "1 << 2"], 
                        ["FLAG_G4SCINTILLATION_COLLECT_PHOTON","1 << 3"], 
                        ["FLAG_G4SCINTILLATION_COLLECT_PROP",  "1 << 4"], 
                        ["FLAG_G4CERENKOV_ADD_SECONDARY",     "1 << 16"], 
                        ["FLAG_G4CERENKOV_KILL_SECONDARY",    "1 << 17"], 
                        ["FLAG_G4CERENKOV_COLLECT_STEP",      "1 << 18"], 
                        ["FLAG_G4CERENKOV_COLLECT_PHOTON",    "1 << 19"], 
                        ["FLAG_G4CERENKOV_APPLY_WATER_QE",    "1 << 20"]
                      ]
    }


"""
import sys, json, logging
log = logging.getLogger(__name__)

class Tmpl(list):
   head = property(lambda self:self.tmpl[0] % dict(kls=self.kls))
   body = property(lambda self:"\n".join(map(lambda _:self.tmpl[1] % dict(kls=self.kls,flag=_[0],bit=_[1]),self)))
   tail = property(lambda self:self.tmpl[2] % dict(kls=self.kls))

   def __init__(self, kls,flags):
       self.kls = kls
       self[:] = flags

   def __str__(self):
       return "\n".join([self.head,self.body,self.tail])


class Enum(Tmpl):
    tmpl=[
r"""
    enum
    {
        FLAG_ZERO = 0,
"""
,
r"""        %(flag)s = %(bit)s, """
,
r"""        
        FLAG_LAST = 1 << 31 
    };
"""
]

class StaticDecl(Tmpl):
    tmpl=["",
r"""    static const char* _%(flag)s ;""",
""]

class StaticImp(Tmpl):
    tmpl=["",
r"""    const char* %(kls)s::_%(flag)-40s = "%(flag)s" ;""",
""]


class FlagsImp(Tmpl):
    tmpl=[r"""
std::string %(kls)s::Flags()
{
    std::vector<std::string> elem ; 
"""
,
r"""    if(HasFlag(%(flag)s)) elem.push_back(std::string(_%(flag)s)) ;"""
,
r"""
    return join(elem, '\n') ; 
}
"""]

class MatchImp(Tmpl):
    tmpl=[r"""
int %(kls)s::MatchFlag(const char* flag )
{
    int ret = FLAG_ZERO ; 
"""
,
r"""    if(strcmp(flag, _%(flag)s ) == 0)  ret = %(flag)s ;"""
,
r"""
    //cout << "%(kls)s::MatchFlag " << flag << " " << ret << endl ;

    return ret ; 
}
"""]

 

def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv)>1:
         path = sys.argv[1]
    else:
         path = "flags.json" 
    pass

    log.info("reading %s " % path )
    js = json.load(file(path))

    for kls in js:    
        flags = js[kls]
        print Enum(kls,flags)
        print StaticDecl(kls,flags)
        print StaticImp(kls,flags)
        print FlagsImp(kls,flags)
        print MatchImp(kls,flags)


if __name__ == '__main__':
    main() 

     





