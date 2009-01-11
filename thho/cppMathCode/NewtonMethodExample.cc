// Newton-Raphson method of finding roots                                     //
//   Passing references to functions f(x) and f'(x) as function parameters    //
//   also demonstrates use of a function template                             //

#include <iostream>
#include <complex>

using namespace std;

//----------------------------------------------------------------------------//
// Function template: Newton-Raphson method find a root of the equation f(x)  //
//  see http://en.wikipedia.org/wiki/Newton's_method                          //
// Parameters in:  &x            reference to first approximation of root     //
//                 (&f)(x)       reference to function f(x)                   //
//                 (fdiv)(x)     reference to function f'(x)                  //
//                 max_loop      maxiumn number of itterations                //
//                 accuracy      required accuracy                            //
//            out: &x            return root found                            //
// function result: > 0 (true) if root found, 0 (false) if max_loop exceeded  //
template <class T1>
 int newton(T1 &x, T1 (&f)(T1), T1 (&fdiv)(T1),
                  int max_loop, const double accuracy)
 {
    T1 term;
    do
        {
         // calculate next term f(x) / f'(x) then subtract from current root  
         term = f(x) / (fdiv)(x);
         x = x - term;                                               // new root
        }
    // check if term is within required accuracy or loop limit is exceeded
    while ((abs(term / x) > accuracy) && (--max_loop));
    cout << "final max_loop " << max_loop << endl;
    return max_loop;
 }
template <class T1>
 int newtonB(T1 &x, T1 (&f)(T1), T1 (&fdiv)(T1),
                  int max_loop, const double accuracy)
 {
    T1 term;
    do
        {
         // calculate next term f(x) / f'(x) then subtract from current root  
         term = f(x) / (fdiv)(x);
         x = x - term;                                               // new root
	 //cout << "\nnow accuracy is " << abs(term/x) << endl;
        }
    // check if term is within required accuracy or loop limit is exceeded
    while ((abs(term / x) > accuracy) && (--max_loop));
    cout << "final max_loop " << max_loop << endl;
    return max_loop;
 }


//----------------------------------------------------------------------------//
// test functions
double func_5(double x)
  {  return x*x - 612 ; }

double fdiv_5(double x) {

    cout << "\nfdiv_5 value is " << 2*x << endl;
    return 2*x ; 

}

double fdiv_5B(double x) {

    double y = (func_5(x+1.0e-12) - func_5(x))/1.0e-12;
    cout << "\nfdiv_5B value is " << y << endl;
    return y;
}

//----------------------------------------------------------------------------//
// Main program to test above function
int main()
{
    cout << "\n\nFind root of x*x -612 = 0\n";
    double x = 10.0;
    if ( newton(x, func_5, fdiv_5, 7, 1.0e-8))
	cout << "\n     root x = " << x << ", test of f(x) = " << func_5(x);
    else cout << "\n    failed to find root ";
    x = 10.0;
    if ( newtonB(x, func_5, fdiv_5B, 7, 1.0e-8))
	cout << "\n     root x = " << x << ", test of f(x) = " << func_5(x);
    else cout << "\n    failed to find root ";


    cin.get();
    return 0;
}


