#!/usr/bin/env python
"""

The 255 confirms suspicion of values truncation::

    In [19]: run material_properties.py
    __dd__Materials__PPE0xa54f978
        ABSLENGTH            : (59)[1.55e-06 0.001 6.2e-06 0.001 1.033e-05 0.001 1.55e-05 0.001] 
    __dd__Materials__MixGas0xa505de8
    __dd__Materials__Air0xa43c470
        ABSLENGTH            : (59)[1.55e-06 1e+07 6.2e-06 1e+07 1.033e-05 1e+07 1.55e-05 1e+07] 
        RINDEX               : (67)[1.55e-06 1.00027 6.2e-06 1.00027 1.033e-05 1.00027 1.55e-05 1.00027] 
    __dd__Materials__Bakelite0xa43e3a0
    __dd__Materials__Foam0xa49e450
    __dd__Materials__Aluminium0xa3fbb40
    __dd__Materials__Iron0xa4607b8
    __dd__Materials__GdDopedLS0xa4cb298
        ABSLENGTH            : (255)[1.3778e-06 299.6 1.3793e-06 306.2 1.3808e-06 328.4 1.3824e-06 363.1 1.3839e-06 385.4 1.3855e-06 386.9 1.387e-06 413.6 1.3886e-06 443.3 1.3901e-06 460 1.3917e-06 463.2 1.3933e-06 450 1.3948e-06 479.9 1.3964e-06 471 1.398e-06 462.3 1.3995e-06 422 1.4011e-06] 
        AlphaFASTTIMECONSTANT : (8)[-1 1 1 1] 
        AlphaSLOWTIMECONSTANT : (10)[-1 35 1 35] 
        AlphaYIELDRATIO      : (14)[-1 0.65 1 0.65] 
        FASTCOMPONENT        : (255)[1.55e-06 0 2.0664e-06 0.001787 2.06985e-06 0.001729 2.07331e-06 0.001969 2.07679e-06 0.002015 2.08027e-06 0.001937 2.08377e-06 0.001676 2.08728e-06 0.00197 2.0908e-06 0.001916 2.09433e-06 0.002049 2.09787e-06 0.001776 2.10143e-06 0.001997 2.10499e-06 0.00] 
        FASTTIMECONSTANT     : (14)[-1 3.64 1 3.64] 
        GammaFASTTIMECONSTANT : (8)[-1 7 1 7] 
        GammaSLOWTIMECONSTANT : (10)[-1 31 1 31] 
        GammaYIELDRATIO      : (16)[-1 0.805 1 0.805] 


"""
import os
import lxml.etree as ET
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"



if __name__ == '__main__':
    xml = parse_("g4_00.dae")
    for mat in  xml.findall(".//{%s}material" % COLLADA_NS ):
        print mat.attrib['id']
        extra = mat.find(".//{%s}extra" % COLLADA_NS ) 

        props = extra.findall(".//{%s}property" % COLLADA_NS )
        data = {} 
        for matrix in extra.findall(".//{%s}matrix" % COLLADA_NS ):
            name, coldim, vals = matrix.attrib['name'],matrix.attrib['coldim'],matrix.attrib['values'] 
            assert coldim == '2'
            data[name] = vals

        for prop in props:
            name, ref = prop.attrib['name'], prop.attrib['ref']
            vals = data[ref] 
            print "    %-20s : (%s)[%s] " % (name, len(vals),vals) 





