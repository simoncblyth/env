
#include "TEveDigitSet_Additions.h"

#include <TRef.h>
#include <TEveDigitSet.h>
#include <TEveUtil.h>
#include <cstdio>

struct DigitBase_t
{
      // Base-class for digit representation classes.

      Int_t fValue; // signal value of a digit (can be direct RGBA color)
      TRef  fId;    // external object reference

      DigitBase_t(Int_t v=0) : fValue(v), fId() {}
      virtual ~DigitBase_t() {}
};


Int_t TEveDigitSet_GetDigitValue(TEveDigitSet* ds, Int_t idx )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   return d->fValue ;
}

void TEveDigitSet_PrintDigit(TEveDigitSet* ds, Int_t idx )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   if (ds->GetValueIsColor()){
      UChar_t* x = (UChar_t*) & d->fValue;
      printf("TEveDigitSet_PrintDigit - 0x%lx, id=%d, r=%d g=%d b=%d a=%d  \n", (ULong_t) ds, idx, x[0], x[1], x[2], x[3] );
   } else {
      printf("TEveDigitSet_PrintDigit - 0x%lx, id=%d, value=%d \n", (ULong_t) ds, idx, d->fValue );
   }
}

void TEveDigitSet_SetDigitValue(TEveDigitSet* ds, Int_t idx , Int_t value )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   d->fValue = value ;
}

void TEveDigitSet_SetDigitColorI(TEveDigitSet* ds, Int_t idx, Color_t ci )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   TEveUtil::ColorFromIdx(ci, (UChar_t*) & d->fValue, kTRUE);
}

/*
 55    static void     ColorFromIdx(Color_t ci, UChar_t col[4], Bool_t alpha=kTRUE);
 56    static void     ColorFromIdx(Color_t ci, UChar_t col[4], Char_t transparency);
 57    static void     ColorFromIdx(Float_t f1, Color_t c1, Float_t f2, Color_t c2,
 58                                 UChar_t col[4], Bool_t alpha=kTRUE);

*/


void TEveDigitSet_SetDigitColorIT(TEveDigitSet* ds, Int_t idx, Color_t ci , Float_t ftrans )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));

  // version cut guess awaits bug reports
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,00)
   TEveUtil::ColorFromIdx(ci, (UChar_t*) & d->fValue, (Char_t)(255*ftrans));
#else
   TEveUtil::ColorFromIdx(ci, (UChar_t*) & d->fValue, (UChar_t)(255*ftrans));
#endif


}

void TEveDigitSet_SetDigitColorRGBA(TEveDigitSet* ds, Int_t idx, Float_t r , Float_t g, Float_t b, Float_t a )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   UChar_t* x = (UChar_t*) & d->fValue;

   x[0] = (UChar_t)(255*r); 
   x[1] = (UChar_t)(255*g); 
   x[2] = (UChar_t)(255*b); 
   x[3] = (UChar_t)(255*a); 

}
