#include <QObject>
#include <QString>

#include "demoifadaptor.h"
#include "mydemo.h"


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    MyDemo* demo = new MyDemo;
    new DemoIfAdaptor(demo);
    QDBusConnection connection = QDBusConnection::sessionBus();
    bool ret = connection.registerService("com.nokia.Demo");
    ret = connection.registerObject("/", demo);    

    return a.exec();
}

