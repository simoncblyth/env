#include <QObject>
#include <QString>
#include <QVariantMap>


class MyDemo : public QObject
{
Q_OBJECT
public:
    explicit MyDemo(QObject *parent = 0);

 
public Q_SLOTS:
    void SayBye();
    void SayHello(const QString &name, const QVariantMap &customdata);
    void SayHelloThere(const QString &name);
Q_SIGNALS:
    void LateEvent(const QString &eventkind);
 
};


