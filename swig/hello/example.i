/* File : example.i */
%module example
%{
/* Put headers and other declarations here ... are copied verbatim  */
extern double My_variable;
extern int    fact(int);
extern int    my_mod(int n, int m);
%}

extern double My_variable;
extern int    fact(int);
extern int    my_mod(int n, int m);


