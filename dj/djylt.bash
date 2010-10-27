# === func-gen- : dj/djylt fgp dj/djylt.bash fgn djylt fgh dj
djylt-src(){      echo dj/djylt.bash ; }
djylt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djylt-src)} ; }
djylt-vi(){       vi $(djylt-source) ; }
djylt-env(){      elocal- ; }
djylt-usage(){
  cat << EOU
     djylt-src : $(djylt-src)
     djylt-dir : $(djylt-dir)

      http://code.google.com/p/django-yui-layout-templates/

      http://developer.yahoo.com/yui/grids/
      http://developer.yahoo.com/yui/fonts/
      http://developer.yahoo.com/yui/reset/
      http://developer.yahoo.com/yui/base/


   * django template inheritance chain via extends
      * layout_n_* <  layout_overrides < layout_base

   * crucial primary body/div id in block '''yahooui''' 
      * layout_base.html:    <div {% block yahooui %}{% endblock yahooui %}>
      * http://developer.yahoo.com/yui/grids/ for what the doc3,doc4,.. etc are  


g4pb:layouts blyth$ grep yahooui *.html
layout_1_column_full_width.html            :  {% block yahooui %}id="doc3"{% endblock yahooui %}
layout_2_columns_narrow_left_column.html   :  {% block yahooui %}id="doc3" class="yui-t2"{% endblock yahooui %}
layout_2_columns_narrow_right_column.html  :  {% block yahooui %}id="doc3" class="yui-t4"{% endblock yahooui %}
layout_2_equal_columns.html                :  {% block yahooui %}id="doc3"{% endblock yahooui %}
layout_3_columns_quarter_half_quarter.html :  {% block yahooui %}id="doc3"{% endblock yahooui %}
layout_3_columns_varying_width.html        :  {% block yahooui %}id="doc3" class="yui-t2"{% endblock yahooui %}
layout_3_equal_columns.html                :  {% block yahooui %}id="doc3"{% endblock yahooui %}
layout_4_equal_columns.html                :  {% block yahooui %}id="doc3"{% endblock yahooui %}



EOU
}
djylt-dir(){ echo $(local-base)/env/dj/djylt/templates ; }
djylt-cd(){  cd $(djylt-dir); }
djylt-mate(){ mate $(djylt-dir) ; }
djylt-url(){ echo http://django-yui-layout-templates.googlecode.com/svn/trunk/ ; }

djylt-get(){
   local dir=$(dirname $(djylt-dir)) &&  mkdir -p $dir && cd $dir
   svn co $(djylt-url) templates
}
