#include <QtDebug>
#include "mydemo.h"


MyDemo::MyDemo(QObject *parent)
{
   qDebug() << "MyDemo::MyDemo" ;
}

void MyDemo::SayHello(const QString &name, const QVariantMap &customdata)
{
   qDebug() << "MyDemo::SayHello" ;
}

void MyDemo::SayBye()
{
   qDebug() << "MyDemo::SayBye" ;
}



