#include <string>
#include "dbxml/DbXml.hpp"

using namespace std;
using namespace DbXml;

int main(int argc, char **argv)
{
    try {
	string tmp = "writer.dbxml" ;
	string dname = "fragment.xml" ;
        cout << "creating container " << tmp << endl ;

        XmlManager mgr;
	XmlContainer cont = mgr.createContainer(tmp) ;  
	XmlUpdateContext uc = mgr.createUpdateContext();
        XmlDocument doc = mgr.createDocument();
        doc.setName(dname);

        XmlEventWriter& writer = cont.putDocumentAsEventWriter(doc, uc);  
        writer.writeStartDocument(NULL, NULL, NULL); // no XML decl

            writer.writeStartElement((const unsigned char *)"a", NULL, NULL, 0, false);

                writer.writeStartElement((const unsigned char *)"b", NULL, NULL, 2, false);
                    writer.writeAttribute((const unsigned char *)"a1", NULL, NULL, (const unsigned char *)"one", true);
                    writer.writeAttribute((const unsigned char *)"b2", NULL, NULL, (const unsigned char *)"two", true);
                    writer.writeText(XmlEventReader::Characters, (const unsigned char *)"b node text", 11);  // 11 is character count, excluding termination 
                writer.writeEndElement((const unsigned char *)"b", NULL, NULL);

                writer.writeStartElement((const unsigned char *)"c", NULL, NULL, 0, false);
                    writer.writeText(XmlEventReader::Characters, (const unsigned char *)"c node text", 11);  // again 11 chars
                writer.writeEndElement((const unsigned char *)"c", NULL, NULL);

            writer.writeEndElement((const unsigned char *)"a", NULL, NULL);

        writer.writeEndDocument();
        writer.close(); 

    } catch (XmlException &e) {
        cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
