// 
//   http://www.gtk.org/api/2.6/glib/glib-Hash-Tables.html
//   http://www.ibm.com/developerworks/library/l-glib.html
//   http://www.ibm.com/developerworks/linux/library/l-glib2.html
//
//     gcc ht.c  -I/opt/local/include/glib-2.0 -I/opt/local/lib/glib-2.0/include  -L/opt/local/lib -lglib-2.0 -o ht
//
//     gcc `glib-config --cflags` -g ht.c -o ht `glib-config --libs`
//
// pkg names for yum/port  ... glib2 glib2-devel 
//     sudo port contents glib2
//
//
//
//   see : /opt/local/include/glib-2.0/glib/ghash.h
//

#include <stdlib.h> 
#include <stdio.h>
#include <glib.h>


static void dump_hash_table_entry(gpointer key, gpointer value, gpointer user_data)
{
    printf("dump_hash_table_entry :  \"%s\" \"%s\" ", key, value);
}

int main(){

   GHashTable *table = g_hash_table_new(g_str_hash, g_str_equal);  // funcs for : hashing, key comparison 
  
   FILE *fp;
   char buf[1024];

   fp = fopen("file.txt", "r");
   if(!fp) exit(1); /* if file does not exist, exit */

   while(fgets(buf, sizeof(buf), fp)) {
       char *key, *value;
       /* get the first and the second field */
       key = strtok(buf, "\t");
       if(!key) continue;
       value = strtok(NULL, "\t");
       if(!value) continue;
	   printf("%s %s", key, value);
       
       // insert (key,value) with replacement and freeing if preexisting key
	   char *old_key, *old_value;
	   if(g_hash_table_lookup_extended(table, key, (gpointer*)&old_key, (gpointer*)&old_value)) {
	       g_hash_table_insert(table, g_strdup(key), g_strdup(value));
	       g_free(old_key);
	       g_free(old_value);
	   } else {
	       g_hash_table_insert(table, g_strdup(key), g_strdup(value));
       }
   
   }
   fclose(fp);

   g_hash_table_foreach(table, dump_hash_table_entry, NULL);

   char* the_key, *the_value ;
   the_key = "one" ;
   the_value = g_hash_table_lookup(table, the_key);
   if(the_value)
   {
	  printf("lookup \"%s\" yields \"%s\" \n", the_key, the_value ) ; 
   }
   else
   {
      printf("lookup \"%s\" yields nowt \n", the_key ) ; 	 
   }

   
   g_hash_table_destroy(table);      

   return 0 ;
}

