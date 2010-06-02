#ifndef TEVEDIGITSET_ADDITIONS_H
#define TEVEDIGITSET_ADDITIONS_H

#include <Rtypes.h>
class TEveDigitSet ;

//#include <TEveDigitSet.h>
//#include <TEveUtil.h>
//#include <cstdio>

Int_t TEveDigitSet_GetDigitValue(TEveDigitSet* ds, Int_t idx );
void TEveDigitSet_PrintDigit(TEveDigitSet* ds, Int_t idx );
void TEveDigitSet_SetDigitValue(TEveDigitSet* ds, Int_t idx , Int_t value );
void TEveDigitSet_SetDigitColorI(TEveDigitSet* ds, Int_t idx, Color_t ci );
void TEveDigitSet_SetDigitColorIT(TEveDigitSet* ds, Int_t idx, Color_t ci , Float_t ftrans );
void TEveDigitSet_SetDigitColorRGBA(TEveDigitSet* ds, Int_t idx, Float_t r , Float_t g, Float_t b, Float_t a );


#endif

