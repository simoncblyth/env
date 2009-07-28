#include <TRef.h>
#include <TEveDigitSet.h>
#include <cstdio>

struct DigitBase_t
{
      // Base-class for digit representation classes.

      Int_t fValue; // signal value of a digit (can be direct RGBA color)
      TRef  fId;    // external object reference

      DigitBase_t(Int_t v=0) : fValue(v), fId() {}
      virtual ~DigitBase_t() {}
};

Int_t TEveDigitSet_GetValue(TEveDigitSet* ds, Int_t idx )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   return d->fValue ;
}

void TEveDigitSet_PrintValue(TEveDigitSet* ds, Int_t idx )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   printf("TEveDigitSet_PrintValue - 0x%lx, id=%d, value=%d \n", (ULong_t) ds, idx, d->fValue );
}

void TEveDigitSet_SetValue(TEveDigitSet* ds, Int_t idx , Int_t value )
{
   DigitBase_t* d = (DigitBase_t*)(ds->GetDigit(idx));
   d->fValue = value ;
}
