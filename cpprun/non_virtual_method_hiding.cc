#include <iostream>

/**

http://stackoverflow.com/questions/11067975/visual-c-overriding-non-virtual-methods


simon:tmp blyth$ cc t.cc -lc++
simon:tmp blyth$ ./a.out 
Base: Non-virtual display.
Base: Virtual display.
Derived: Non-virtual display.
Derived: Virtual display.
simon:tmp blyth$ 


**/


using namespace std;

class Base
{
public:
    void Display(){ cout << "Base::Display" << endl ; }
    virtual void vDisplay(){ cout<<"Base: Virtual display."<<endl; }

    void Other(){ cout << "Base::Other" << endl ; }
};

class Derived : public Base
{
public:

    void Display(){    cout << "Derived::Display" << endl ; }
    virtual void vDisplay(){ cout<<"Derived::vDisplay."<<endl; }

};

int main()
{
    Base ba;
    Derived de;

    ba.Display();
    ba.vDisplay();
    ba.Other();

    de.Display();
    de.vDisplay();
    de.Other();

    return 0;
};
