"""
   Tutorial from 
      http://docs.python-rum.org/user/tutorial.html

"""
import datetime

from sqlalchemy import Column, ForeignKey, Table, PrimaryKeyConstraint
from sqlalchemy.types import *
from sqlalchemy.orm import relation
from sqlalchemy.ext.declarative import declarative_base

Model = declarative_base()

class Person(Model):
    __tablename__ = "person"

    id = Column('id', Integer, primary_key=True)
    name = Column('name', Unicode, nullable=False)
    type = Column('type', String(50), nullable=False)
    age = Column('age', Integer)

    __mapper_args__ = {
        'polymorphic_on':type,
        'polymorphic_identity': 'Person'
        }

    def __unicode__(self):
        return self.name


class Actor(Person):
    __tablename__ = "actor"

    id = Column('id', Integer, ForeignKey('person.id'), primary_key=True)
    oscars_won = Column('oscars_won', Integer, default=0)
    __mapper_args__ = {
        'polymorphic_identity': 'Actor',
        }


class Director(Person):
    __tablename__ = "director"

    id = Column('id', Integer, ForeignKey('person.id'), primary_key=True)
    chairs_broken = Column('chairs_broken', Integer, default=0)
    __mapper_args__ = {
        'polymorphic_identity': 'Director',
        }


class Genre(Model):
    __tablename__ = "genre"

    id = Column('id', Integer, primary_key=True)
    name = Column('name', Unicode, nullable=False)
    
    def __unicode__(self):
        return self.name

# This is a database table we won't map to a class since it's only an
# implementation detail required to create many-to-many associations in a
# relational database.
_actor_movie = Table('actor_movie', Model.metadata,
    Column('actor_id', Integer, ForeignKey('actor.id')),
    Column('movie_id', Integer, ForeignKey('movie.id')),
    PrimaryKeyConstraint('movie_id', 'actor_id'),
    )


class Movie(Model):
    __tablename__ = "movie"

    id = Column('id', Integer, primary_key=True)
    title = Column('title', Unicode, nullable=False)
    filmed_on = Column('filmed_on', Date)
    genre_id = Column('genre_id', Integer, ForeignKey('genre.id'))
    director_id = Column('director_id', Integer, ForeignKey('director.id'))
    synopsis = Column('synopsis', Unicode)
    
    genre = relation('Genre', backref='movies')
    director = relation('Director', backref='movies')
    actors = relation('Actor', secondary=_actor_movie, backref='movies')

    def __unicode__(self):
        ret = self.title
        if self.filmed_on:
            ret += " (%d)" % self.filmed_on.year
        return ret

class Rental(Model):
    __tablename__ = "rental"
    id = Column('id', Integer, primary_key=True)
    person_id = Column('person_id', Integer, ForeignKey('person.id'),
                       nullable=False)
    movie_id = Column('movie_id', Integer, ForeignKey('movie.id'),
                      nullable=False)
    date = Column('date', DateTime)
    due_date = Column('due_date', DateTime)

    movie = relation('Movie', backref='rentals')
    person = relation('Person', backref='rentals')

    def is_overtime(self):
        return self.due_date > datetime.datetime.now()

    def __unicode__(self):
        return u"%s -> %s" % (self.movie, self.person)




if __name__=='__main__':
    import model
    from sqlalchemy import create_engine

    from sqlalchemy.orm import create_session
    engine = create_engine('sqlite:///movie.db')
    model.Model.metadata.create_all(engine)    ## model.Model is the declarative base of all the objects
    session = create_session(bind=engine)
    actor = model.Actor(name=u'Sean Jollystar')
    director = model.Director(name=u'Lucy Jollygood')
    genre = model.Genre(name=u'Comedy')
    movie = model.Movie(title=u'Jolly Summer', director=director,
    actors=[actor] , genre=genre )
    session.add(movie) 
    session.flush()
    session.clear()
    print session.query(model.Person).all()


