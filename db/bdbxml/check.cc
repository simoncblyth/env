/*

   g++ -c -I$BDBXML_HOME/include check.cc
   g++    -L$BDBXML_HOME/lib -ldbxml check.o -o check
   ./check      # reruns give error when pre-existing container


*/
#include <string>
#include <fstream>
#include "dbxml/DbXml.hpp"

using namespace std;
using namespace DbXml;

int main(int argc, char **argv)
{
    try {
        XmlManager mgr;
        
        // Create the phonebook container
        XmlContainer cont = mgr.createContainer("check.dbxml");
        
        // Add the phonebook entries to the container
        XmlUpdateContext uc = mgr.createUpdateContext();
        cont.putDocument("phone1", "<phonebook><name><first>Tom</first><last>Jones</last></name><phone type=\"home\">420-203-2032</phone></phonebook>", uc);
        cont.putDocument("phone2", "<phonebook><name><first>Lisa</first><last>Smith</last></name><phone type=\"home\">420-992-4801</phone><phone type=\"cell\">390-812-4292</phone></phonebook>", uc);

        // Run an XQuery against the phonebook container
        XmlQueryContext qc = mgr.createQueryContext();        
        XmlResults res = 
        mgr.query("collection('check.dbxml')/phonebook[name/first = 'Lisa']/phone[@type = 'home']/string()", qc);

        // Print out the result of the query
        XmlValue value;
        while (res.next(value)) cout << "Value: " << value.asString() << endl;

    } catch (XmlException &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
