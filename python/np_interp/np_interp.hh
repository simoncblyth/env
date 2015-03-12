#ifndef NPINTERP_H
#define NPINTERP_H

#include "stdio.h"
#include "assert.h"


struct Ary 
{
    double* values ;
    unsigned int length ;

    Ary(double* _values, unsigned int _length) : values(_values), length(_length) {}
    Ary(unsigned int _length) : length(_length) 
    { 
        values = new double[length];
    }

    double getLeft(){  return values[0] ; }
    double getRight(){ return values[length-1] ; }
    double getValue(unsigned int index){ return values[index] ;}
    double* getValues(){ return values ; }

    int binary_search(double key)
    {
        if(key > values[length-1])
        {
            return length ;
        }
        unsigned int imin = 0 ; 
        unsigned int imax = length ; 
        unsigned int imid ;   

        while(imin < imax)
        {
            imid = imin + ((imax - imin) >> 1);
            if (key >= values[imid]) 
            {
                imin = imid + 1;
            }
            else 
            {
                imax = imid;
            }
        }
        return imin - 1; 
    } 
};


Ary* np_interp(Ary* xi, Ary* xp, Ary* fp )
{
    //
    // Loosely follow np.interp signature and implementation from 
    //    https://github.com/numpy/numpy/blob/v1.9.1/numpy/lib/src/_compiled_base.c#L599
    //

    assert(xp->length == fp->length);  // input domain and values must be of same length

    // input domain and values
    double* dx = xp->getValues();   
    double* dy = fp->getValues();   
    double left = fp->getLeft();
    double right = fp->getRight();

    Ary* res = new Ary(xi->length); // Ary to be filled with interpolated values
    double* dres = res->getValues();

    for (unsigned int i = 0; i < res->length ; i++) 
    {
        const double z = xi->getValue(i);
        int j = xp->binary_search(z);

        if(j == -1)
        {
            dres[i] = left;
        }
        else if(j == xp->length - 1)
        {
            dres[i] = dy[j];
        }
        else if(j == xp->length )
        {
            dres[i] = right;
        }
        else
        {
            const double slope  = (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j]);
            dres[i] = slope*(z - dx[j]) + dy[j];
        }
    }
    return res ;
}




struct Prop
{
   Ary* x ;
   Ary* y ;

   Prop(Ary* _x, Ary* _y) : x(_x), y(_y) 
   {
       assert(x->length == y->length);
   }
   Prop(unsigned int n)
   {
       x = new Ary(n);
       y = new Ary(n);
   }

   unsigned int getLength()
   {
       return x->length ; 
   }

   void setXY(unsigned int index, double _x, double _y )
   { 
       double* xv = x->getValues();
       double* yv = y->getValues();
       xv[index] = _x ;  
       yv[index] = _y ;  
   }

   void setX(unsigned int index, double _x )
   { 
       double* xv = x->getValues();
       xv[index] = _x ;  
   }
   void setY(unsigned int index, double _y )
   { 
       double* yv = y->getValues();
       yv[index] = _y ;  
   }


   void interpolateY(Prop* src)
   {
       y = np_interp( x, src->x, src->y ); 
   } 

   void dump(const char* msg)
   {
       printf("%s\n", msg );
       for(unsigned int i=0 ; i < getLength() ; i++)
       {
          printf(" %2u  %10.3f %10.3f \n", i, x->getValue(i), y->getValue(i)); 
       }
   }


};


#endif
