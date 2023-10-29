from typing import List, Tuple
import psycopg2.errors
from psycopg2 import sql
from psycopg2.sql import SQL, Literal, Identifier, Composed

import Utility.DBConnector as Connector
from Utility.DBConnector import ResultSet
from Utility.ReturnValue import ReturnValue
from Utility.Exceptions import DatabaseException

from Business.Movie import Movie
from Business.Studio import Studio
from Business.Critic import Critic
from Business.Actor import Actor

# ---------------------------------- CRUD API: ----------------------------------

create_table_dict = {
    'Critics': 'CREATE TABLE Critics('
               '    cId INT PRIMARY KEY CHECK (cId > 0),'
               '    name VARCHAR NOT NULL'
               ');',

    'Movies': 'CREATE TABLE Movies ('
              '  name VARCHAR,'
              '  year INT CHECK (year >= 1895),'
              'genre VARCHAR(6) CHECK (genre =\'Drama\' OR genre =\'Action\' OR genre =\'Comedy\' OR genre =\'Horror\'),'
              '  PRIMARY KEY (name,year)'
              ');',

    'Actors': 'CREATE TABLE Actors ('
              '     aId INT PRIMARY KEY check(aId > 0),'
              '     name VARCHAR NOT NULL ,'
              '     age INT NOT NULL CHECK ( age > 0 ),'
              '     height INT NOT NULL CHECK ( height > 0 )'
              ');',

    'Studios': 'CREATE TABLE Studios ('
               '    sId INT PRIMARY KEY CHECK (sId > 0),'
               '    name VARCHAR NOT NULL'
               ');',

    'MovieCritics': 'CREATE TABLE MovieCritics ('
                    '   movie_name VARCHAR NOT NULL,'
                    '   movie_year INT NOT NULL ,'
                    '   cId INT NOT NULL,'
                    '   rating INT NOT NULL CHECK ( rating >= 1 and rating <= 5 ),'
                    '   FOREIGN KEY (movie_name, movie_year) REFERENCES Movies(name, year) ON DELETE CASCADE,'
                    '   FOREIGN KEY (cId) REFERENCES Critics(cId) ON DELETE CASCADE,'
                    '   UNIQUE (movie_year, movie_name, cId)'
                    ');',
    'MovieActors': 'CREATE TABLE MovieActors ('
                   '    movie_name VARCHAR NOT NULL,'
                   '    movie_year INT NOT NULL,'
                   '    aId INT NOT NULL,'
                   '    salary INT NOT NULL CHECK ( salary > 0 ) ,'
                   '    role_played VARCHAR(100) NOT NULL ,'
                   '    FOREIGN KEY (movie_name, movie_year) REFERENCES Movies(name, year) ON DELETE CASCADE,'
                   '    FOREIGN KEY (aId) REFERENCES Actors(aId) ON DELETE CASCADE,'
                   '    UNIQUE(movie_name,movie_year,aId,role_played)'
                   ');',

    'Productions': 'CREATE TABLE Productions ('
                   '    movie_name VARCHAR NOT NULL,'
                   '    movie_year INT NOT NULL,'
                   '    sId INT NOT NULL,'
                   '    budget INT NOT NULL CHECK (budget >= 0),'
                   '    revenue INT NOT NULL CHECK (revenue >= 0),'
                   '    FOREIGN KEY (movie_name, movie_year) REFERENCES Movies(name, year) ON DELETE CASCADE,'
                   '    FOREIGN KEY (sId) REFERENCES Studios(sId) ON DELETE CASCADE,'
                   '    UNIQUE(movie_name, movie_year)'
                   ');',

    'AverageRating': 'CREATE VIEW AverageRating AS '
                     '  SELECT genre, movie_name,movie_year, avg(rating) as avg_rating '
                     '  FROM moviecritics mc '
                     '  JOIN movies m on mc.movie_name = m.name and mc.movie_year = m.year '
                     '  group by movie_name, movie_year, genre;',

    'ActorsMoviesNoRole': 'CREATE VIEW ActorsMoviesNoRole AS '
                          ' SELECT movie_name, movie_year,aid,salary '
                          ' FROM movieactors '
                          ' GROUP BY movie_name, movie_year,aid,salary;',

    'avgRatingActors': 'CREATE VIEW avgRatingActors AS '
                       '    SELECT ma.aid, ma.movie_name, ma.movie_year, avg_rating, ag.genre'
                       '    FROM averagerating ag '
                       '    JOIN ActorsMoviesNoRole ma ON ag.movie_year = ma.movie_year AND ag.movie_name = ma.movie_name '
                       'UNION '
                       'SELECT ma.aid, ma.movie_name, ma.movie_year, 0 as avg_rating, m.genre '
                       'FROM moviecritics ag '
                       'RIGHT OUTER JOIN ActorsMoviesNoRole ma ON ag.movie_year = ma.movie_year AND ag.movie_name = ma.movie_name '
                       'JOIN movies m on ma.movie_year = m.year and ma.movie_name = m.name '
                       'WHERE ag.rating IS NULL;',

    'actorStudio': 'CREATE VIEW actorStudio AS'
                   '    SELECT aid, sid '
                   '    FROM ActorsMoviesNoRole am '
                   '    JOIN productions p on p.movie_name = am.movie_name and p.movie_year = am.movie_year;',

    'MovieProdActor': 'CREATE VIEW movieprodactors AS'
                      ' select m.name, m.year, budget, sum(salary) as sum_salaries '
                      ' from movies m '
                      ' LEFT OUTER join productions p on m.name = p.movie_name and m.year = p.movie_year '
                      ' LEFT OUTER join actorsmoviesnorole a on p.movie_name = a.movie_name and p.movie_year = a.movie_year'
                      ' group by m.name, m.year, budget;'

}


def result_set_2_list(result: ResultSet):
    return result.rows


def createTables():
    conn = None
    try:
        conn = Connector.DBConnector()
        conn.execute("".join(create_table_dict.values()))
    except DatabaseException.ConnectionInvalid as e:
        print(e)
    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
    except DatabaseException.FOREIGN_KEY_VIOLATION as e:
        print(e)
    except Exception as e:
        print(e)
    finally:
        # will happen any way after try termination or exception handling
        conn.close()


def clearTables():
    cnx = None
    query = 'DELETE FROM Critics;DELETE FROM Movies;DELETE FROM Actors;DELETE FROM Studios;DELETE FROM MovieCritics;DELETE FROM MovieActors;DELETE FROM Productions;'
    try:
        cnx = Connector.DBConnector()
        cnx.execute(query)

    except Exception as e:
        print(e)
        return ReturnValue.ERROR

    finally:
        cnx.close()
        return ReturnValue.ERROR


def dropTables():
    delete_tables_str = 'drop view movieprodactors; drop view avgRatingActors; drop view averageRating; drop view actorStudio;  drop view ActorsMoviesNoRole; drop table moviecritics;drop table critics;drop table movieactors;drop table actors;drop table productions;drop table movies;drop table studios;'
    conn = None

    try:
        conn = Connector.DBConnector()
        conn.execute(delete_tables_str)
    except DatabaseException.ConnectionInvalid as e:
        print(e)
    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
    except DatabaseException.FOREIGN_KEY_VIOLATION as e:
        print(e)
    except Exception as e:
        print(e)
    finally:
        # will happen any way after try termination or exception handling
        conn.close()


def addCritic(critic: Critic) -> ReturnValue:
    conn = None
    ret_val = None
    try:
        conn = Connector.DBConnector()

        cId = Literal(critic.getCriticID())
        name = Literal(critic.getName())

        query = sql.SQL("INSERT INTO Critics(cid, name) VALUES({i}, {n})").format(i=cId, n=name)

        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_val = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.ALREADY_EXISTS

    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_val


def deleteCritic(critic_id: int) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_val = None
    try:
        conn = Connector.DBConnector()

        cid = Literal(critic_id)

        query = SQL("DELETE FROM critics WHERE cid = {cid}").format(cid=cid)
        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK

    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_val = ReturnValue.NOT_EXISTS
        return ret_val


def getCriticProfile(critic_id: int) -> Critic:
    cid = Literal(critic_id)
    query = SQL("SELECT cid, name FROM critics WHERE cid = {cid}").format(cid=cid)
    cnx = None
    effected_rows, result = 0, ResultSet()
    try:
        cnx = Connector.DBConnector()
        effected_rows, result = cnx.execute(query)
    except Exception as e:
        print(e)
        return Critic.badCritic()

    finally:

        cnx.close()
        if effected_rows == 0:
            return Critic.badCritic()
        else:
            result = result_set_2_list(result)[0]
            return Critic(*result)


def addActor(actor: Actor) -> ReturnValue:
    conn = None
    ret_val = None
    try:
        conn = Connector.DBConnector()

        aid = Literal(actor.getActorID())
        name = Literal(actor.getActorName())
        age = Literal(actor.getAge())
        height = Literal(actor.getHeight())
        query = sql.SQL("INSERT INTO Actors VALUES({aid}, {name}, {age}, {height})").format(
            aid=aid, name=name, age=age, height=height)
        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_val = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.ALREADY_EXISTS

    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_val


def deleteActor(actor_id: int) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_val = None

    try:
        conn = Connector.DBConnector()

        aid = Literal(actor_id)

        query = SQL("DELETE FROM actors WHERE aid = {aid}").format(aid=aid)
        rows_effected, _ = conn.execute(query)
        ret_value = ReturnValue.OK

    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_value = ReturnValue.NOT_EXISTS
        return ret_value


def getActorProfile(actor_id: int) -> Actor:
    aid = Literal(actor_id)
    query = SQL("SELECT * FROM actors WHERE aid = {aid}").format(aid=aid)
    cnx = None
    effected_rows, result = 0, ResultSet()
    try:
        cnx = Connector.DBConnector()
        effected_rows, result = cnx.execute(query)
    except Exception as e:
        print(e)
        return Actor.badActor()

    finally:

        cnx.close()
        if effected_rows == 0:
            return Actor.badActor()
        else:
            result = result_set_2_list(result)[0]
            return Actor(*result)


def addMovie(movie: Movie) -> ReturnValue:
    conn = None
    ret_val = None
    try:
        conn = Connector.DBConnector()

        name = Literal(movie.getMovieName())
        year = Literal(movie.getYear())
        genre = Literal(movie.getGenre())

        query = sql.SQL("INSERT INTO Movies VALUES({name}, {year}, {genre})").format(year=year, name=name, genre=genre)

        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_val = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.ALREADY_EXISTS

    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_val


def deleteMovie(movie_name: str, year: int) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_val = None
    try:
        conn = Connector.DBConnector()

        name = Literal(movie_name)
        year = Literal(year)

        query = SQL("DELETE FROM movies WHERE name={name} and year={year}").format(name=name, year=year)
        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK

    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_val = ReturnValue.NOT_EXISTS
        return ret_val


def getMovieProfile(movie_name: str, year: int) -> Movie:
    name = Literal(movie_name)
    year = Literal(year)

    query = SQL("SELECT * FROM movies WHERE name = {name} AND year = {year}").format(name=name, year=year)
    cnx = None
    effected_rows, result = 0, ResultSet()
    try:
        cnx = Connector.DBConnector()
        effected_rows, result = cnx.execute(query)
    except Exception as e:
        print(e)
        return Movie.badMovie()

    finally:

        cnx.close()
        if effected_rows == 0:
            return Movie.badMovie()
        else:
            result = result_set_2_list(result)[0]
            return Movie(*result)


def addStudio(studio: Studio) -> ReturnValue:
    conn = None
    ret_val = None
    try:
        conn = Connector.DBConnector()

        sid = Literal(studio.getStudioID())
        name = Literal(studio.getStudioName())

        query = sql.SQL("INSERT INTO Studios(sid, name) VALUES({sid}, {name})").format(sid=sid, name=name)

        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_val = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_val = ReturnValue.ALREADY_EXISTS

    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_val


def deleteStudio(studio_id: int) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_val = None
    try:
        conn = Connector.DBConnector()

        sid = Literal(studio_id)

        query = SQL("DELETE FROM studios WHERE sid = {sid}").format(sid=sid)
        rows_effected, _ = conn.execute(query)
        ret_val = ReturnValue.OK
    except Exception as e:
        print(e)
        ret_val = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_val = ReturnValue.NOT_EXISTS
        return ret_val


def getStudioProfile(studio_id: int) -> Studio:
    sid = Literal(studio_id)
    query = SQL("SELECT * FROM studios WHERE sid = {sid}").format(sid=sid)
    cnx = None
    effected_rows, result = 0, ResultSet()
    try:
        cnx = Connector.DBConnector()
        effected_rows, result = cnx.execute(query)
    except Exception as e:
        print(e)
        return Studio.badStudio()

    finally:

        cnx.close()
        if effected_rows == 0:
            return Studio.badStudio()
        else:
            result = result_set_2_list(result)[0]
            return Studio(*result)


def criticRatedMovie(movieName: str, movieYear: int, criticID: int, rating: int) -> ReturnValue:
    conn = None
    ret_value = None
    try:
        conn = Connector.DBConnector()

        cId = Literal(criticID)
        movie_name = Literal(movieName)
        movie_year = Literal(movieYear)
        rate = Literal(rating)

        query = sql.SQL("INSERT INTO MovieCritics VALUES({name}, {year}, {cid}, {rate})").format(name=movie_name
                                                                                                 , year=movie_year
                                                                                                 , cid=cId
                                                                                                 , rate=rate)
        rows_effected, _ = conn.execute(query)
        ret_value = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_value = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.ALREADY_EXISTS

    except DatabaseException.FOREIGN_KEY_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.NOT_EXISTS

    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_value


def criticDidntRateMovie(movieName: str, movieYear: int, criticID: int) -> ReturnValue:
    """

    :rtype: object
    """
    conn = None
    rows_effected = 0
    ret_value = None

    try:
        conn = Connector.DBConnector()

        m_name = Literal(movieName)
        m_year = Literal(movieYear)
        cId = Literal(criticID)

        query = SQL("DELETE FROM MovieCritics WHERE cId = {cid} and movie_name = {name} and movie_year = {year}") \
            .format(cid=cId
                    , name=m_name
                    , year=m_year)
        rows_effected, _ = conn.execute(query)
        ret_value = ReturnValue.OK

    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_value = ReturnValue.NOT_EXISTS
        return ret_value


def actorPlayedInMovie(movieName: str, movieYear: int, actorID: int, salary: int, roles: List[str]) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_value = None

    if roles == []:
        return ReturnValue.BAD_PARAMS
    try:
        conn = Connector.DBConnector()

        m_name = Literal(movieName)
        m_year = Literal(movieYear)
        aId = Literal(actorID)
        salary = Literal(salary)
        check_exists_query = SQL("SELECT * FROM movieactors "
                                 "WHERE aId = {aId} and movie_name = {m_n} and movie_year = {m_y}").format(aId=aId,
                                                                                                           m_n=m_name,
                                                                                                           m_y=m_year)

        n, _ = conn.execute(check_exists_query)

        if n != 0:
            ret_value = ReturnValue.ALREADY_EXISTS

        else:
            query_values = []
            for role in roles:
                query_values.append(
                    SQL("({m_name}, {m_year}, {aId}, {salary}, {role})").format(m_name=m_name, m_year=m_year, aId=aId,
                                                                                salary=salary, role=Literal(role)))
            query_values = SQL(",").join(query_values)

            query = SQL("INSERT INTO movieactors VALUES {query_values} ;").format(query_values=query_values)
            rows_effected, _ = conn.execute(query)

            ret_value = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_value = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.ALREADY_EXISTS

    except DatabaseException.FOREIGN_KEY_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.NOT_EXISTS

    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_value


def getActorsRoleInMovie(actor_id: int, movie_name: str, movieYear: int):
    conn = None
    rows_effected = 0
    ret_value = []

    try:
        conn = Connector.DBConnector()

        m_name = Literal(movie_name)
        m_year = Literal(movieYear)
        aId = Literal(actor_id)
        query = SQL("SELECT role_played "
                    "FROM movieactors "
                    "WHERE aid = {aid} AND movie_name = {m_name} AND movie_year = {m_year} ORDER BY role_played DESC "
                    ).format(aid=aId, m_year=m_year, m_name=m_name)
        row_effected, result = conn.execute(query)
        ret_value = [c[0] for c in result_set_2_list(result)]

    except Exception as e:
        print('exept')
        ret_value = []

    finally:
        return ret_value


def actorDidntPlayInMovie(movieName: str, movieYear: int, actorID: int) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_value = None
    try:
        conn = Connector.DBConnector()

        m_name = Literal(movieName)
        m_year = Literal(movieYear)
        aId = Literal(actorID)

        query = SQL("DELETE FROM MovieActors WHERE aId = {aid} and movie_name = {name} and movie_year = {year}") \
            .format(aid=aId,
                    name=m_name,
                    year=m_year)
        rows_effected, _ = conn.execute(query)
        ret_value = ReturnValue.OK
    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_value = ReturnValue.NOT_EXISTS
        return ret_value


def studioProducedMovie(studioID: int, movieName: str, movieYear: int, budget: int, revenue: int) -> ReturnValue:
    conn = None
    ret_value = None
    try:
        conn = Connector.DBConnector()

        sId = Literal(studioID)
        m_name = Literal(movieName)
        m_year = Literal(movieYear)
        budget = Literal(budget)
        revenue = Literal(revenue)

        query = sql.SQL("INSERT INTO Productions VALUES({name}, {year}, {sid}, {budg}, {rev})").format(name=m_name,
                                                                                                       year=m_year,
                                                                                                       sid=sId,
                                                                                                       budg=budget,
                                                                                                       rev=revenue)
        rows_effected, _ = conn.execute(query)
        ret_value = ReturnValue.OK

    except DatabaseException.ConnectionInvalid as e:
        print(e)
        ret_value = ReturnValue.ERROR

    except DatabaseException.NOT_NULL_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.BAD_PARAMS

    except DatabaseException.CHECK_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.BAD_PARAMS

    except DatabaseException.UNIQUE_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.ALREADY_EXISTS

    except DatabaseException.FOREIGN_KEY_VIOLATION as e:
        print(e)
        ret_value = ReturnValue.NOT_EXISTS

    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        return ret_value


def studioDidntProduceMovie(studioID: int, movieName: str, movieYear: int) -> ReturnValue:
    conn = None
    rows_effected = 0
    ret_value = None
    try:
        conn = Connector.DBConnector()

        m_name = Literal(movieName)
        m_year = Literal(movieYear)
        sId = Literal(studioID)

        query = SQL("DELETE FROM productions WHERE sId = {sid} and movie_name = {name} and movie_year = {year}") \
            .format(sid=sId,
                    name=m_name,
                    year=m_year)
        rows_effected, _ = conn.execute(query)
        ret_value = ReturnValue.OK
    except Exception as e:
        print(e)
        ret_value = ReturnValue.ERROR
    finally:
        conn.close()
        if rows_effected == 0:
            ret_value = ReturnValue.NOT_EXISTS
        return ret_value


# ---------------------------------- BASIC API: ----------------------------------
def averageRating(movieName: str, movieYear: int) -> float:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        movieName = Literal(movieName)
        movieYear = Literal(movieYear)
        query = SQL("SELECT avg_rating "
                    "FROM averageRating "
                    "WHERE movie_name = {m_n} and movie_year = {m_y};").format(m_n=movieName, m_y=movieYear)
        row_effected, result = conn.execute(query)

    except Exception as e:
        print(e)
        return 0

    finally:
        conn.close()
        if row_effected == 1:
            return result_set_2_list(result)[0][0]
        else:
            return 0


def averageActorRating(actorID: int) -> float:
    conn = None
    row_effected, result = 0, None
    ret_val = 0
    try:
        conn = Connector.DBConnector()
        aid = Literal(actorID)
        query = SQL("SELECT avg(avg_rating) FROM avgRatingActors WHERE aid = {aid};").format(aid=aid)
        row_effected, result = conn.execute(query)

    except Exception as e:
        print(e)
        ret_val = 0

    finally:
        conn.close()
        if row_effected == 1:
            ret_val = result_set_2_list(result)[0][0]
        if ret_val is None:
            ret_val = 0
        return ret_val


def bestPerformance(actor_id: int) -> Movie:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        aid = Literal(actor_id)
        query = SQL("SELECT movie_name,movie_year, genre "
                    "   from avgRatingActors "
                    "   where aid = {aid} and avg_rating >= ALL"
                    "       (SELECT avg_rating FROM avgRatingActors where aid = {aid}) "
                    "   ORDER BY movie_year, movie_name DESC"
                    "   limit 1;").format(aid=aid)
        row_effected, result = conn.execute(query)

    except Exception as e:
        print(e)
        return Movie.badMovie()

    finally:
        if row_effected == 1:
            return Movie(*result_set_2_list(result)[0])
        else:
            return Movie.badMovie()


def stageCrewBudget(movieName: str, movieYear: int) -> int:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        movieName = Literal(movieName)
        movieYear = Literal(movieYear)
        query = SQL("SELECT budget - sum_salaries as stage_budget "
                    "FROM movieprodactors "
                    "WHERE budget IS NOT NULL and sum_salaries IS NOT NULL "
                    "AND name = {m_n}  AND year = {m_y} "
                    "UNION "
                    "SELECT 0 as stage_budget "
                    "FROM movieprodactors "
                    "WHERE budget IS NULL "
                    "AND name = {m_n}  AND year = {m_y} "
                    "UNION "
                    "SELECT budget as stage_budget "
                    "FROM movieprodactors "
                    "WHERE sum_salaries IS NULL and budget IS NOT NULL "
                    "AND name = {m_n}  AND year = {m_y};").format(m_n=movieName, m_y=movieYear)
        row_effected, result = conn.execute(query)

    except Exception as e:
        print(e)
        return -1
    finally:
        if row_effected == 0:
            return -1
        else:
            return result_set_2_list(result)[0][0]


def overlyInvestedInMovie(movie_name: str, movie_year: int, actor_id: int) -> bool:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        movieName = Literal(movie_name)
        movieYear = Literal(movie_year)
        aid = Literal(actor_id)
        query = SQL("SELECT n_roles_actors.cnt * 2 >= n_roles_movie.cnt "
                    "FROM "
                    "   (SELECT aid, movie_name, movie_year, count(role_played) as cnt "
                    "   FROM movieactors GROUP BY aid, movie_name, movie_year) n_roles_actors "
                    "JOIN "
                    "   (SELECT movie_name, movie_year, count(role_played) as cnt "
                    "   FROM movieactors "
                    "   GROUP BY movie_name, movie_year) n_roles_movie "
                    "On n_roles_movie.movie_name = n_roles_actors.movie_name and n_roles_movie.movie_year = n_roles_actors.movie_year "
                    "Where n_roles_actors.movie_year = {m_y} and n_roles_actors.movie_name = {m_n} and aid = {aid};"
                    ).format(m_n=movieName, m_y=movieYear, aid=aid)
        row_effected, result = conn.execute(query)

    except Exception as e:
        print(e)
        return False

    finally:

        conn.close()
        if row_effected == 1:
            return result_set_2_list(result)[0][0]
        else:

            return False


# ---------------------------------- ADVANCED API: ----------------------------------


def franchiseRevenue() -> List[Tuple[str, int]]:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        query = SQL("SELECT m.name as m_name, sum(revenue) as movie_revenue "
                    "FROM movies m "
                    "JOIN productions p on p.movie_name = m.name and p.movie_year = m.year "
                    "GROUP BY m.name "
                    "UNION "
                    " (SELECT m.name as m_name, 0 as movie_revenue"
                    " FROM movies m "
                    " LEFT JOIN productions p on p.movie_name = m.name and p.movie_year = m.year "
                    " GROUP BY m.name "
                    "HAVING SUM(revenue) IS NULL )"
                    "order by m_name desc ;")
        row_effected, result = conn.execute(query)
    except Exception as e:
        print(e)
        return []
    finally:
        conn.close()
        return result_set_2_list(result)


def studioRevenueByYear() -> List[Tuple[str, int]]:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        query = SQL("SELECT sid, movie_year, sum(revenue)"
                    " FROM productions "
                    "group by sid, movie_year "
                    "order by sid desc, movie_year desc;")
        row_effected, result = conn.execute(query)
    except Exception as e:
        print(e)
        return []
    finally:
        conn.close()
        return result_set_2_list(result)


def getFanCritics() -> List[Tuple[int, int]]:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        query = SQL("select cid, A.sid "
                    "FROM "
                    "   (SELECT sid, count(*) as prod_cnt "
                    "   FROM productions p "
                    "   GROUP BY sid) A "
                    "JOIN "
                    "   (SELECT cid, sid, count(*) as critic_on_studio_cnt "
                    "   FROM moviecritics mc "
                    "   JOIN productions p on mc.movie_year = p.movie_year and mc.movie_name = p.movie_name "
                    "   GROUP BY cid, sid) B ON A.sid = B.sid "
                    "WHERE A.prod_cnt = B.critic_on_studio_cnt"
                    " ORDER BY cid DESC, sid DESC;")

        row_effected, result = conn.execute(query)
    except Exception as e:
        print(e)
        return []
    finally:
        conn.close()
        return result_set_2_list(result)


def averageAgeByGenre() -> List[Tuple[str, float]]:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        query = SQL("SELECT genre, AVG(age) "
                    "FROM ( SELECT distinct a.aid, genre, age "
                    "       FROM actorsmoviesnorole as am "
                    "       JOIN actors a on am.aid = a.aid "
                    "       JOIN movies m on movie_year = m.year and movie_name = m.name) B "
                    "group by genre "
                    "ORDER BY genre;")
        row_effected, result = conn.execute(query)
    except Exception as e:
        print(e)
        return []
    finally:
        conn.close()
        return result_set_2_list(result)


def getExclusiveActors() -> List[Tuple[int, int]]:
    conn = None
    row_effected, result = 0, None
    try:
        conn = Connector.DBConnector()
        query = SQL("SELECT distinct aid, sid "
                    "FROM actorStudio "
                    "WHERE aid IN "
                    "   (SELECT aid "
                    "   FROM actorStudio "
                    "   GROUP BY aid "
                    "   having count(distinct sid) = 1) "
                    "ORDER BY aid DESC;")
        row_effected, result = conn.execute(query)
    except Exception as e:
        print(e)
        return []
    finally:
        conn.close()
        return result_set_2_list(result)


# GOOD LUCK!


if __name__ == '__main__':
    dropTables()
    createTables()

    addMovie(Movie('m1', 2000, 'Drama'))
    addMovie(Movie('m1', 2001, 'Drama'))

    addStudio(Studio(1, 's1'))

    studioProducedMovie(1, 'm1', 2000, 1, 1)

    print(franchiseRevenue())

    """addActor(Actor(1, 'actor_1', 10, 160))
    addActor(Actor(2, 'actor_2', 20, 160))
    addActor(Actor(3, 'actor_3', 15, 160))

    addMovie(Movie('SpiderMan', 2000, 'Drama'))
    addMovie(Movie('SpiderMan', 2005, 'Drama'))
    addMovie(Movie('SpiderMan2', 2000, 'Action'))
    addMovie(Movie('AntMan', 2000, 'Action'))

    actorPlayedInMovie('SpiderMan', 2000, 1, 1000, ['parker', 'peter'])
    actorPlayedInMovie('SpiderMan2', 2000, 1, 1000, ['gween'])
    actorPlayedInMovie('SpiderMan', 2000, 2, 1000, ['gros'])

    addStudio(Studio(1, 'studio_1'))
    addStudio(Studio(2, 'studio_2'))
    addStudio(Studio(3, 'studio_3'))

    studioProducedMovie(1, 'SpiderMan', 2000, 1000, 1000)
    studioProducedMovie(3, 'SpiderMan', 2005, 1000, 1234)
    studioProducedMovie(2, 'SpiderMan2', 2000, 1000, 1000)

    addCritic(Critic(1, 'critic_1'))
    addCritic(Critic(2, 'critic_2'))
    addCritic(Critic(3, 'critic_1'))
    addCritic(Critic(4, 'critic_2'))

    criticRatedMovie('SpiderMan', 2000, 1, 1)

    criticRatedMovie('SpiderMan', 2000, 2, 3)
    criticRatedMovie('SpiderMan', 2000, 3, 5)

    criticRatedMovie('SpiderMan2', 2000, 1, 1)
    criticRatedMovie('SpiderMan2', 2000, 2, 2)
    criticRatedMovie('SpiderMan2', 2000, 3, 3)

    addMovie(Movie('LaRue', 2000, 'Drama'))
    studioProducedMovie(1, 'LaRue', 2000, 1000, 10000)

    addMovie(Movie('LaRue', 2003, 'Comedy'))
    studioProducedMovie(1, 'LaRue', 2003, 1000, 10000)

    actorPlayedInMovie('LaRue', 2003, 3, 3000, ['r1', "r2"])
    actorPlayedInMovie('LaRue', 2003, 1, 3000, ['r4', "RE", 'R5'])

    criticRatedMovie('LaRue', 2000, 2, 1)
    criticRatedMovie('LaRue', 2003, 2, 1)

    print(averageRating('SpiderMan', 2000))"""
