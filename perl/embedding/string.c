/*
    cc -o string string.c `perl -MExtUtils::Embed -e ccopts -e ldopts`

*/
#include <EXTERN.h>
#include <perl.h>

static PerlInterpreter *my_perl;

main (int argc, char **argv, char **env)
{
    char *embedding[] = { "", "-e", "0" };

    PERL_SYS_INIT3(&argc,&argv,&env);
    my_perl = perl_alloc();
    perl_construct( my_perl );

    perl_parse(my_perl, NULL, 3, embedding, NULL);
    PL_exit_flags |= PERL_EXIT_DESTRUCT_END;
    perl_run(my_perl);

    /** Treat $a as an integer **/
    eval_pv("$a = 3; $a **= 2", TRUE);
    printf("a = %d\n", SvIV(get_sv("a", 0)));

    /** Treat $a as a float **/
    eval_pv("$a = 3.14; $a **= 2", TRUE);
    printf("a = %f\n", SvNV(get_sv("a", 0)));

    /** Treat $a as a string **/
    eval_pv("$a = 'rekcaH lreP rehtonA tsuJ'; $a = reverse($a);", TRUE);
    printf("a = %s\n", SvPV_nolen(get_sv("a", 0)));


    /** better way without pollution **/
    SV *val = eval_pv("reverse 'rekcaH lreP rehtonA tsuJ'", TRUE);
    printf("%s\n", SvPV_nolen(val));


    perl_destruct(my_perl);
    perl_free(my_perl);
    PERL_SYS_TERM();
}
