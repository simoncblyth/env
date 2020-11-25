// gcc std_future_references.cpp -std=c++11 -lstdc++ -o /tmp/std_future_references && /tmp/std_future_references
// https://stackoverflow.com/questions/41041707/why-cant-i-use-reference-in-stdfuture-parameters

/**

Like std::thread, std::async passes the parameters by value to the "function".
If you have a function that takes a reference you need to wrap the variable you
are passing to asyc with std::ref like

If the function takes a const & then you need to use std::cref.


Consider what would happen if it did bind bar by reference.

Then every time you called std::async, every value you passed would have to
last until the async completed.

That would be a recipe for accidental memory corruption. So, std::async instead
by default copies everything you pass to it.

It then runs the task on the copy of your input.

Being smart, it tells the code you are calling that the value is non-persistent
by moving it into the code. And lvalue references cannot bind to moved-from
values.

You can override this behavior by using std::reference_wrapper.  async
understands reference_wrapper, and it automatically stores a reference to those
values and passes them by-reference to the called code.

The easy way to create a reference_wrapper is to call std::ref.


**/


#include <future>
#include <iostream>
#include <string>
#include <cassert>

int main()
{
    int foo = 0;
    bool bar = false;

    auto fn = [=, &foo](bool& out) -> std::string { out = true ; return "str " ; };

    std::future<std::string> async_request = std::async(std::launch::async, fn, std::ref(bar));
    std::cout << async_request.get() << std::endl;

    assert( bar == true );

    return 0 ; 
}
