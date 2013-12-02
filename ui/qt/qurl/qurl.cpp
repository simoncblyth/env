#include <QDebug>
#include <QStringList>
#include <QString>
#include <QUrl>
#include <QPair>
#include <QList>
#include <QChar>
#include <QMap>

#include <iostream>

int main(){

    QUrl url("http://localhost/dae/tree/3199.html?c=1,0,0&a=1,1,1&fov=50");

    qDebug() << "host [" << url.host() << "]" ; 
    qDebug() << "path [" << url.path() << "]" ; 

    QString path(url.path());
    QStringList elems(path.split("/"));
    QString name(elems.at(elems.size()-1)); 
    qDebug() << "name [" << name << "]" ; 

    int index(0) ;

    // eg split 3199.html into 3199 and html 
    QStringList basetype(name.split("."));
    if( basetype.size() == 2 ){
        QString base(basetype.at(0)); 
        QString type(basetype.at(1)); 
        qDebug() << "base [" << base << "]" ; 
        qDebug() << "type [" << type << "]" ; 
        index = base.toInt();
    }
    qDebug() << "index [" << index << "]" ; 


    QMap<QString, QString> kt;  
    kt["c"] = "v" ; kt["cam"] = "v" ;
    kt["a"] = "v" ; kt["at"] = "v" ;
    kt["u"] = "v" ; kt["up"] = "v" ;
    kt["n"] = "f" ; kt["near"] = "f" ;
    kt["r"] = "f" ; kt["far"] = "f" ;
    kt["m"] = "i" ; kt["mode"] = "i" ;
    kt["o"] = "s" ; kt["orth"] = "s" ;

    QMapIterator<QString, QString> ikt(kt);
    while (ikt.hasNext()) {
        ikt.next();
        qDebug() << ikt.key() << ": " << ikt.value() ;
    }


    typedef QPair<QString, QString>   Qpss ;  
   
    foreach (Qpss kv, url.queryItems())
    {
        qDebug() << "------------------------------------- " ; 
        QString k(kv.first) ;
        QString v(kv.second) ;
        qDebug() << "[" << k << "] : [" << v << "] " ;    
        std::cout << "[" << k.toStdString() << "] : [" << v.toStdString() << "] " << std::endl ; 

        QString ktype = kt.value(k,"");
        if(ktype.startsWith("v"))
        {
            QStringList vals = v.split(",");  
            if(vals.size() == 3)
            {
                float x = vals.at(0).toFloat();    
                float y = vals.at(1).toFloat();    
                float z = vals.at(2).toFloat();    
                qDebug("  xyz  %f %f %f ",x,y,z);
            }
            else if(vals.size() == 1)
            {
                float x = vals.at(0).toFloat();    
                float y = x ; 
                float z = x ;    
                qDebug("  xyz  %f %f %f ",x,y,z);
            }  

        } else if(ktype.startsWith("f")) {

            float fval = v.toFloat(); 
            qDebug() << "val " << k << " : " << fval ; 

        } else if(ktype.startsWith("i")) {

            int ival = v.toInt(); 
            qDebug() << "ival " << k << " : " << ival ; 

        } else if(ktype.startsWith("s")) {

            QString sval(v); 
            qDebug() << "sval " << k << " : " << sval ; 

        } 
    }

    return 0 ;
}

