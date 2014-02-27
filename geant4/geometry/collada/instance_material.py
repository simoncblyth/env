#!/usr/bin/env python
"""

::

    (chroma_env)delta:collada blyth$ ./instance_material.py ../materials/g4_00.dae.6 0
    [0 ] Bakelite                  : #__dd__Materials__Bakelite0xab630b0 
    [1 ] BPE                       : #__dd__Materials__BPE0xae88760 
    [2 ] LiquidScintillator        : #__dd__Materials__LiquidScintillator0xab59dc8 
    [3 ] ADTableStainlessSteel     : #__dd__Materials__ADTableStainlessSteel0xae72758 
    [4 ] Acrylic                   : #__dd__Materials__Acrylic0xae4be78 
    [5 ] GdDopedLS                 : #__dd__Materials__GdDopedLS0xab86340 
    [6 ] Teflon                    : #__dd__Materials__Teflon0xaa8ade8 
    [7 ] MixGas                    : #__dd__Materials__MixGas0xab25110 
    [8 ] PPE                       : #__dd__Materials__PPE0xaba1b48 
    [9 ] RadRock                   : #__dd__Materials__RadRock0xb2cd1d8 
    [10] Foam                      : #__dd__Materials__Foam0xab24098 
    [11] OpaqueVacuum              : #__dd__Materials__OpaqueVacuum0xab62ab8 
    [12] Water                     : #__dd__Materials__Water0xae9c3c0 
    [13] PVC                       : #__dd__Materials__PVC0xaa94a28 
    [14] Vacuum                    : #__dd__Materials__Vacuum0xaf1d298 
    [15] NitrogenGas               : #__dd__Materials__NitrogenGas0xae12b08 
    [16] Silver                    : #__dd__Materials__Silver0xae0fd70 
    [17] UnstStainlessSteel        : #__dd__Materials__UnstStainlessSteel0xaa73b60 
    [18] Tyvek                     : #__dd__Materials__Tyvek0xab26538 
    [19] ESR                       : #__dd__Materials__ESR0xaeaaeb8 
    [20] StainlessSteel            : #__dd__Materials__StainlessSteel0xadf7930 
    [21] Bialkali                  : #__dd__Materials__Bialkali0xaf1b7e8 
    [22] Air                       : #__dd__Materials__Air0xab09580 
    [23] Iron                      : #__dd__Materials__Iron0xaa66250 
    [24] Rock                      : #__dd__Materials__Rock0xab06f88 
    [25] DeadWater                 : #__dd__Materials__DeadWater0xaabb308 
    [26] Nylon                     : #__dd__Materials__Nylon0xab72bf8 
    [27] OwsWater                  : #__dd__Materials__OwsWater0xabb2118 
    [28] Aluminium                 : #__dd__Materials__Aluminium0xaa65b70 
    [29] Co_60                     : #__dd__Materials__Co_600xae0d998 
    [30] Pyrex                     : #__dd__Materials__Pyrex0xae6d0e0 
    [31] Ge_68                     : #__dd__Materials__Ge_680xae9a758 
    [32] C_13                      : #__dd__Materials__C_130xae0f438 
    [33] MineralOil                : #__dd__Materials__MineralOil0xaecfd78 
    [34] Nitrogen                  : #__dd__Materials__Nitrogen0xab09148 
    [35] IwsWater                  : #__dd__Materials__IwsWater0xab82978 


"""
import os, sys
from collada.xmlutil import ET, COLLADA_NS

tag = lambda _:str(ET.QName(COLLADA_NS,_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()

if __name__ == '__main__':
    xml = parse_(sys.argv[1])
    verbosity = int(sys.argv[2]) 

    materials = {}
    library_materials = xml.find(".//" + tag("library_materials"))
    for material in library_materials.findall(tag("material")):
        materials[material.attrib['id']] = material

    dmap = {}
    for im in xml.findall(".//" + tag("instance_material")):
        symbol = im.attrib['symbol']
        target = im.attrib['target'][1:]  # remove '#' prefix
        if symbol in dmap:
            assert dmap[symbol] == target
        else:
            dmap[symbol] = target 
        pass


    if verbosity == 0:
        print "\n".join(["[%-2s] %-25s : %s " % (i,symbol,target) for i,(symbol,target) in enumerate(dmap.items())]) 
    elif verbosity == 1:
        print "\n".join(["[%-2s] %-25s : %s \n %s " % (i,symbol,target,ET.tostring(materials[target])) for i,(symbol,target) in enumerate(dmap.items())]) 





