/*
   clang -x c++ -lstdc++ np_interp.cc && ./a.out && rm a.out
   clang++ np_interp.cc && ./a.out && rm a.out
*/

#include "np_interp.hh"

int main(int argc, char** argv)
{
    Prop* src = new Prop(5);
    src->setXY(0, 0.f,  0.f );   
    src->setXY(1, 1.f, 10.f );   
    src->setXY(2, 2.f, 20.f );   
    src->setXY(3, 3.f, 30.f );   
    src->setXY(4, 4.f, 40.f );   
    src->dump("src");

    Prop* dst = new Prop(3);
    dst->setX(0, 0.5f) ;
    dst->setX(1, 1.5f) ;
    dst->setX(2, 2.5f) ;

    dst->interpolateY(src);
    dst->dump("dst");

    return 0;
}


