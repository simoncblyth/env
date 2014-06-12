#!/usr/bin/env python
"""
http://codingmess.blogspot.tw/2009/05/conversion-of-wavelength-in-nanometers.html

"""

wav2RGB_glsl = r"""
vec4 wav2color(in float wav)
{
    //if (isinf(wav))      w = -1 ;
    //else if( isnan(wav)) w = -2 ;  
    //else     

    //if ( w < 0 ) return vec4( 1., 1., 1., 1. ); 

    int w = int(wav) ;

    vec3 col ;
 
    if      (w >= 380 && w < 440)  col = vec3( -(wav - 440.) / (440. - 350.) , 0.0, 1.0 ) ;
    else if (w >= 440 && w < 490)  col = vec3( 0.0, (wav - 440.) / (490. - 440.), 1.0 ) ;
    else if (w >= 490 && w < 510)  col = vec3( 0.0, 1.0, -(wav - 510.) / (510. - 490.) ) ;
    else if (w >= 510 && w < 580)  col = vec3( (wav - 510.) / (580. - 510.), 1.0 , 0.0 ) ;
    else if (w >= 580 && w < 645)  col = vec3( 1.0,  -(wav - 645.) / (645. - 580.), 0.0 );
    else if (w >= 645 && w <= 780) col = vec3( 1.0, 0.0, 0.0 ) ;
    else                           col = vec3( 0.0, 0.0, 0.0 );

    // intensity correction
    float SSS ;
    if ( w >= 380 && w < 420 )     SSS = 0.3 + 0.7*(wav - 350.) / (420. - 350.);
    else if (w >= 420 && w <= 700) SSS = 1.0 ;
    else if (w > 700 && w <= 780)  SSS = 0.3 + 0.7*(780. - wav) / (780. - 700.);
    else SSS = 0.0 ;

    return vec4( SSS*col, 1.) ; 
}
"""

def wav2RGB(wavelength):
    try:
        w = int(wavelength)
    except OverflowError:    # getting some float infinities
        w = 1000 

    # colour
    if w >= 380 and w < 440:
        R = -(w - 440.) / (440. - 350.)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.) / (490. - 440.)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.) / (510. - 490.)
    elif w >= 510 and w < 580:
        R = (w - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.) / (645. - 580.)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # intensity correction
    if w >= 380 and w < 420:
        SSS = 0.3 + 0.7*(w - 350) / (420 - 350)
    elif w >= 420 and w <= 700:
        SSS = 1.0
    elif w > 700 and w <= 780:
        SSS = 0.3 + 0.7*(780 - w) / (780 - 700)
    else:
        SSS = 0.0

    #SSS *= 255
    #return [int(SSS*R), int(SSS*G), int(SSS*B)]
    return [SSS*R, SSS*G, SSS*B, 1.]

if __name__ == '__main__':
    for w in range(200,800):
        print w, wav2RGB(w)




