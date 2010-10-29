= Overview dybprj =

'''dybprj''' is an umbrella django project 
    * via AuthUserFileBackend provides extrapolated Trac/SVN Users
    * '''TRY TO KEEP PROJECT TO MINIMUM ... BETTER TO WORK IN APPS'''

== logging ==

Use LogEntry from the admin 
   * http://stackoverflow.com/questions/868295/django-admin-action-in-1-1



== ISSUES ==

  * socket/exchange/queue cleanup
      * need to create some torture tests to see if cleanup is effective 

  * alternatives to YUI for presentation css/grids/reset/  etc..
      * http://jqueryui.com/

== POSSIBILITIES ==

  * expose url for comments
  * comments by user with links
  * comments by date with links



== NOTES ==


Note the interleaving of templates ... 

  * obj view templates extend the standard YUI templates from djylt-  with 
{{{
{% extends 'layouts/layout_2_equal_columns.html' %}
}}}     

  * but all those standard layouts extend in turn :
{{{
{% extends 'layout_overrides.html' %}
}}}

 * and the overrides in turn ...
{{{
{% extends 'layouts/layout_base.html' %}
}}}



All those come with the '''djylt''' app  

BUT can impinge onto every layout without touching '''djylt''' via 
  * {{{dybprj/templates/layout_overrides.html}}}

By virtue of the ordering of '''settings.TEMPLATE_DIRS''' with '''dybprj/templates''' ahead of '''djylt''' :
{{{

TEMPLATE_DIRS = (
    PROJECT_DIR + "/templates" , 
    DJYLT_DIR + "/templates" ,
)

}}}


