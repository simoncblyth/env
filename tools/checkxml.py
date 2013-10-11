#!/usr/bin/env python
"""

Various Checks on XML files
===========================


Checking duplicated id/name attributes
----------------------------------------

::

     checkxml.py --dupe --att id   $LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae
     checkxml.py --dupe --att name $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml


::

    simon:~ blyth$ checkxml.py --att name $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml
    2013-10-10 20:04:10,053 env.tools.checkxml INFO     /Users/blyth/env/bin/checkxml.py --att name /usr/local/env/geant4/geometry/gdml/g4_01.gdml
    2013-10-10 20:04:10,055 env.tools.checkxml INFO     reading /usr/local/env/geant4/geometry/gdml/g4_01.gdml 
    2013-10-10 20:04:10,341 env.tools.checkxml INFO     xml findall .//*[@name] 
    elements with name attributes: 17831 : distinct id 17831 distinct jd 17727 (with appended pointer removed) 
    dupid: 0 

    dupjd: 36 
       [3] TopRefGap-ChildForTopRefGapCutHols           ##### names for subtraction and union solids are not unique
       [3] BotReflector-ChildForBotRefHols 
       [3] near_pool_dead-ChildFornear_pool_dead_box 
       [3] TopReflector-ChildForTopRefCutHols 
       [9] cylinder+ChildForsource-assy 
       [2] LsoOflTnkTub1+ChildForLsoOflTnk 
       [3] near_pool_curtain-ChildFornear_pool_curtain_box 
       [11] near_pool_ows-ChildFornear_pool_ows_box 
       [5] GdsOflTnkTub1+ChildForGdsOflTnk 
       [6] BotESR-ChildForBotESRCutHols 
       [2] AcrylicCylinder 
       [5] BlackCylinder-ChildForRadialShieldUnit 
       [2] Turntable-ChildForturntable 
       [7] TopESR-ChildForTopESRCutHols 
       [4] OflTnkContainterTub1+ChildForOflTnkContainer 
       [3] near_pool_liner-ChildFornear_pool_liner_box 
       [3] SstTopCirRibPri-ChildForSstTopCirRibBase 
       [3] near_top_cover-ChildFornear_top_cover_box 
       [4] table_panel-ChildFortable_panel_box 
       [3] BotRefGap-ChildForBotRefGapCutHols 
       [11] near_pool_iws-ChildFornear_pool_iws_box 
       [9] led-cylinder+ChildForled-source-assy 
       [9] amcco60-cylinder+ChildForamcco60-source-assy 
       [3] near-radslab-box-9-box-ChildFornear-radslab-box-9 

       [2] /dd/Materials/Gd_160           ######  these are same name on "element" and "material" elements 
       [2] /dd/Materials/Gd_152 
       [2] /dd/Materials/Iron 
       [2] /dd/Materials/Aluminium 
       [2] /dd/Materials/Gd_155 
       [2] /dd/Materials/Gd_154 
       [2] /dd/Materials/Gd_157 
       [2] /dd/Materials/Gd_156 
       [2] /dd/Materials/Bor_10 
       [2] /dd/Materials/Bor_11 
       [2] /dd/Materials/Nitrogen 
       [2] /dd/Materials/Gd_158 



Check id/name characters
-------------------------

::

     checkxml.py --char --att id   $LOCAL_BASE/env/geant4/geometry/xdae/g4_01.dae
     checkxml.py --char --att name $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml




"""
import os, sys, logging
log = logging.getLogger(__name__)

#import xml.etree.cElementTree as ET
#import xml.etree.ElementTree as ET
import lxml.etree as ET

#COLLADA_NS='http://www.collada.org/2005/11/COLLADASchema'
#tag = lambda _:str(ET.QName(COLLADA_NS,_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
tostring_ = lambda _:ET.tostring(_)
isorted_ = lambda d,idx:sorted(d.items(),key=lambda kv:d[kv[0]].meta[idx]) 


def check_duplicated_id(xml, opts):
    """
    Check if document id are distinct without the pointer appendage 
    """

    ids = []
    jds = []

    allid=set()
    alljd=set()

    dupid=set()
    dupjd=set()

    jdup = {}

    ecount = 0 
    att = opts.att
    xpath = './/*[@' + att + ']'
    log.info("xml findall %s " % xpath )
    for elem in xml.findall(xpath):
        ecount += 1
        id = elem.attrib[att]
        if id[-9:-7] == '0x':
            jd = id[:-9]
        else:
            jd = id

        ids.append(id) 
        jds.append(jd) 

        if id in allid:
           dupid.add(id)
           dup[id] = tostring_(elem)

        if jd in alljd:
           dupjd.add(jd)
           if jd not in jdup:
               jdup[jd] = []
           jdup[jd].append(tostring_(elem))

        allid.add(id)
        alljd.add(jd)
    print "elements with %s attributes: %s : distinct id %s distinct jd %s (with appended pointer removed) " % (att, ecount, len(allid), len(alljd))     
           
    nxd_ = lambda xds,xd:len(filter(lambda _:_ == xd, xds))

    print "dupid: %s " % len(dupid)
    print "\n".join(["   [%s] %s " % (nxd_(ids,_),_) for _ in list(dupid)])

    print "dupjd: %s " % len(dupjd)
    print "\n".join(["   [%s] %s " % (nxd_(jds,_),_) for _ in list(dupjd)])
    for jd in list(dupjd):
        # xpath = ".//*[@"+att+"='"+jd+"']"  not working 
        print "######### ", jd 
        for _ in jdup[jd]:
            print _ 

    #assert ecount == len(allid) == len(alljd)


def check_characters_id(xml, opts):
    att = opts.att
    xpath = './/*[@' + att + ']'
    log.info("xml findall %s " % xpath )
    for elem in xml.findall(xpath):
        id = elem.attrib[att]
        print id
        #if ':' in id:   # ':' always preceded digits in GDML
        #    print id
        #if id[0] == '/' or id[0] == '_':
        #    if '-' in id:    # '-' only in solid named not volume paths
        #        print id
        #    if '.' in id:    # '-' only in solid named not volume paths
        #        print id
  


class Defaults(object):
    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    loglevel = "INFO"
    att = 'id'
    dupe = False
    char = False

def parse_args(doc):
    from optparse import OptionParser
    defopts = Defaults()
    op = OptionParser(usage=doc)
    op.add_option("-f", "--logformat", default=defopts.logformat )
    op.add_option("-l", "--loglevel",   default=defopts.loglevel, help="logging level : INFO, WARN, DEBUG ... Default %default"  )
    op.add_option("-a", "--att",   default=defopts.att)
    op.add_option( "--dupe",  action="store_true",  default=defopts.dupe)
    op.add_option( "--char",  action="store_true",  default=defopts.char)

    opts, args = op.parse_args()
    level = getattr( logging, opts.loglevel.upper() )
    logging.basicConfig(format=opts.logformat,level=level)
    log.info(" ".join(sys.argv))

    return opts, args


def main():
    opts, args = parse_args(__doc__) 
    path = args[0]
    log.info("reading %s " % path )
    xml = parse_(path)

    if opts.dupe:
        check_duplicated_id(xml, opts)
    if opts.char:
        check_characters_id(xml, opts)


if __name__ == '__main__':
    main()


