/**

https://stackoverflow.com/questions/47702776/how-to-properly-delete-pointers-when-using-abstract-classes

A basic C++ rule says that destructors work their way up from the derived class
to the base class. When a Cat is destroyed, then the Cat part is destroyed
first and the Animal part is destroyed after.

delete animal; is undefined behaviour because in order to properly follow C++
destruction rules, one must know, at runtime, which derived class part should
be destroyed before the Animal base part. A virtual destructor does exactly
that - it enables a dynamic dispatch mechanism that makes sure destruction
works as designed.

You have no virtual destructor, however, so delete animal just doesn't make
sense. There is no way to call the correct derived-class destructor, and
destroying only the Animal part wouldn't exactly be meaningful behaviour,
either.

Therefore, the C++ language makes no assumptions about what will happen in such
a situation.

Your compiler is nice enough to warn you about this.

With delete cat, the situation is slightly different. The static type of the
cat pointer is Cat*, not Animal*, so it is clear even without any dynamic
dispatch mechanism which derived-class destructor to call first.

The compiler still warns you about this, but it does so with a different
wording ("might cause" vs. "will cause"). I believe the reason is that Cat
might itself be the base class for more derived classes, seeing as it is
already part of a class hierarchy with virtual functions.

It apparently doesn't bother to execute a more complete code analysis to find
out that delete cat is really harmless.

In order to fix this, make the Animal destructor virtual. While you're at it,
replace your raw pointers with std::unique_ptr. You still have to follow the
virtual destructor rule for classes like yours, but you no longer have to
perform a manual delete.  Share


answered Dec 7, 2017 at 20:16
Christian Hackl's user avatar
Christian Hackl

**/


#include <iostream>

class Animal
{
public:
    virtual void makeNoise() = 0;

    void eat()
    {
        std::cout << "Eating..." << "\n";
    }

    void sleep()
    {
        std::cout << "Sleeping..." << "\n";
    }

#ifdef WITH_FIX
    virtual ~Animal(){} ; 
#endif

};

class Cat: public Animal
{
public:
    void makeNoise()
    {
        std::cout << "Miow..." << "\n";
    }
};

class Cow: public Animal
{
public:
    void makeNoise()
    {
        std::cout << "Mooo..." << "\n";
    }
};

int main()
{
    Animal *animal;
    Cat *cat = new Cat();
    Cow *cow = new Cow();

    animal = cat;
    animal->eat();
    animal->sleep();
    animal->makeNoise();

    animal = cow;
    animal->eat();
    animal->sleep();
    animal->makeNoise();

    delete cat ; 
    delete animal ; 

    return 0;
}
/**

A[blyth@localhost delete-non-virtual-dtor]$ ./animal.sh
animal.cc: In function ‘int main()’:
animal.cc:60:5: error: deleting object of polymorphic class type ‘Cat’ which has non-virtual destructor might cause undefined behavior [-Werror=delete-non-virtual-dtor]
   60 |     delete cat ;
      |     ^~~~~~~~~~
animal.cc:61:5: error: deleting object of abstract class type ‘Animal’ which has non-virtual destructor will cause undefined behavior [-Werror=delete-non-virtual-dtor]
   61 |     delete animal ;
      |     ^~~~~~~~~~~~~
cc1plus: some warnings being treated as errors
A[blyth@localhost delete-non-virtual-dtor]$ 


**/



