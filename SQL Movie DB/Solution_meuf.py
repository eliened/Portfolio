from typing import List, Tuple
from psycopg2 import sql

import Utility.DBConnector as Connector
from Utility.ReturnValue import ReturnValue
from Utility.Exceptions import DatabaseException

from Business.Movie import Movie
from Business.Studio import Studio
from Business.Critic import Critic
from Business.Actor import Actor


# ---------------------------------- CRUD API: ----------------------------------

def createTables():
    # TODO: implement
    conn = None
    try:
        conn = Connector.DBConnector()
        conn.execute("CREATE TABLE Critic(CriticID INTEGER UNIQUE NOT NULL,Name TEXT NOT NULL,"
                     "CHECK (CriticID>0))")
        conn.execute("CREATE TABLE Movie(MovieName TEXT NOT NULL ,Year INTEGER NOT NULL ,Genre TEXT NOT NULL , \
                        UNIQUE (Year,MovieName), CHECK (Year>=1895) , \
                        CHECK (Genre = 'Drama' OR Genre = 'Action' OR Genre = 'Comedy' OR Genre = 'Horror'))")
        conn.execute("CREATE TABLE Actor(ActorID INTEGER UNIQUE NOT NULL,Name TEXT NOT NULL, \
                        Age INTEGER NOT NULL, Height INTEGER NOT NULL, \
                        CHECK(ActorID>0 AND Age>0 AND Height>0))")
        conn.execute("CREATE TABLE Studio(StudioID INTEGER UNIQUE NOT NULL,Name Text NOT NULL,"
                     "CHECK (StudioID>0))")
        conn.execute("CREATE TABLE Rates(MovieName TEXT, Year INTEGER ,"
                     "CriticID INTEGER , Rating INTEGER, "
                     "UNIQUE (MovieName,Year,CriticID),"
                     "FOREIGN KEY(MovieName,Year) REFERENCES Movie(MovieName,Year) ON DELETE CASCADE,"
                     "FOREIGN KEY(CriticID) REFERENCES Critic(CriticID) ON DELETE CASCADE,"
                     "CHECK(1<=Rating AND Rating<=5))")
        conn.execute("CREATE TABLE Plays(MovieName TEXT, Year INTEGER, ActorID INTEGER, Salary INTEGER NOT NULL , "
                     "Roles TEXT[] NOT NULL,"
                     "UNIQUE (MovieName,Year,ActorID),"
                     "FOREIGN KEY(MovieName,Year) REFERENCES Movie(MovieName,Year) ON DELETE CASCADE,"
                     "FOREIGN KEY(ActorID) REFERENCES Actor(ActorID) ON DELETE CASCADE,"
                     "CHECK(Salary>0 AND CARDINALITY(Roles)>0))")
        conn.execute("CREATE OR REPLACE FUNCTION check_roles_not_null()"
                     "RETURNS TRIGGER "
                     "AS $$ "
                     "BEGIN "
                     "IF (SELECT count(*) FROM unnest(NEW.Roles) x WHERE x IS NULL) > 0 THEN "
                     "RAISE EXCEPTION 'Roles array contains null values' USING ERRCODE = '23514';" #fits to cgeackException
                     "END IF;"
                     "RETURN NEW;"
                     "END;"
                     "$$ LANGUAGE plpgsql;"
                     "CREATE TRIGGER check_roles_not_null_trigger "
                     "AFTER INSERT OR UPDATE ON Plays "
                     "FOR EACH ROW "
                     "EXECUTE PROCEDURE check_roles_not_null();")
        conn.execute("CREATE TABLE Produces(StudioID INTEGER ,MovieName TEXT, Year INTEGER,"
                     "Budget INTEGER , Revenue INTEGER, "
                     "UNIQUE (MovieName,Year)," # one movie can be produced by only 1 studio
                     "FOREIGN KEY(StudioID) REFERENCES Studio(StudioID) ON DELETE CASCADE,"
                     "FOREIGN KEY(MovieName,Year) REFERENCES Movie(MovieName,Year) ON DELETE CASCADE,"
                     "CHECK(Budget>=0 AND Revenue >=0))")
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
    pass


def clearTables():
    conn = None
    try:
        conn = Connector.DBConnector()
        conn.execute("TRUNCATE TABLE Critic CASCADE")
        conn.execute("TRUNCATE TABLE Movie CASCADE")
        conn.execute("TRUNCATE TABLE Actor CASCADE")
        conn.execute("TRUNCATE TABLE Studio CASCADE")

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
    pass


def dropTables():
    conn = None
    try:
        conn = Connector.DBConnector()
        drop_query = sql.SQL("DROP VIEW IF EXISTS overlyInvestedInMovie")
        conn.execute(drop_query)
        conn.execute("DROP TABLE IF EXISTS Produces,Plays,Rates")
        conn.execute("DROP TABLE IF EXISTS Studio,Actor,Movie,Critic")

    except DatabaseException.ConnectionInvalid as e:
        # do stuff
        print(e)
    except DatabaseException.NOT_NULL_VIOLATION as e:
        # do stuff
        print(e)
    except DatabaseException.CHECK_VIOLATION as e:
        # do stuff
        print(e)
    except DatabaseException.UNIQUE_VIOLATION as e:
        # do stuff
        print(e)
    except DatabaseException.FOREIGN_KEY_VIOLATION as e:
        # do stuff
        print(e)
    except Exception as e:
        print(e)
    finally:
        # will happen any way after code try termination or exception handling
        conn.close()

    pass


# -------------functions for avoiding code duplicate-----------------#
# -------------macro_api-----------------#

def add_to_table_macro(query):
    """gets a sql literal Add query and executes it, returns Return value"""
    conn = None
    try:
        conn = Connector.DBConnector()
        rows_effected, _ = conn.execute(query)
    except (DatabaseException.NOT_NULL_VIOLATION, DatabaseException.CHECK_VIOLATION,) as e:
        return ReturnValue.BAD_PARAMS
    except (DatabaseException.UNIQUE_VIOLATION, DatabaseException.FOREIGN_KEY_VIOLATION) as e:
        return ReturnValue.ALREADY_EXISTS
    except (DatabaseException.ConnectionInvalid, DatabaseException.UNKNOWN_ERROR, Exception) as e:
        print("sol print: ", e)
        return ReturnValue.ERROR
    finally:
        conn.close()
    return ReturnValue.OK


def delete_from_table_macro(query):
    """gets an sql literal delete query and executes it, returns Return value"""
    conn = None
    try:
        conn = Connector.DBConnector()
        rows_effected, _ = conn.execute(query)
        if rows_effected == 1:
            return ReturnValue.OK
        else:
            return ReturnValue.NOT_EXISTS
    except (DatabaseException.NOT_NULL_VIOLATION, DatabaseException.CHECK_VIOLATION, \
            DatabaseException.UNIQUE_VIOLATION, DatabaseException.FOREIGN_KEY_VIOLATION) as e:
        return ReturnValue.NOT_EXISTS
    except (DatabaseException.ConnectionInvalid, DatabaseException.UNKNOWN_ERROR, Exception) as e:
        return ReturnValue.ERROR
    finally:
        conn.close()
    pass


def get_from_table_macro(query):
    """gets a sql literal "get" query and executes it, returns entries-if succeed or ReturnValue.ERROR if failed"""
    conn = None
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return entries
    except Exception as e:
        return ReturnValue.ERROR
    finally:
        conn.close()
    pass


# -------------functions for critic-----------------#

def addCritic(critic: Critic) -> ReturnValue:
    query = sql.SQL("INSERT INTO Critic(CriticID, Name) VALUES({id}, {cName})").format(
        id=sql.Literal(critic.getCriticID()), cName=sql.Literal(critic.getName()))
    return add_to_table_macro(query)
    pass


def getCriticProfile(critic_id: int) -> Critic:
    query = sql.SQL("SELECT CriticID,name FROM Critic WHERE CriticID = {id}").format(
        id=sql.Literal(critic_id))

    entries = get_from_table_macro(query)
    try:
        return Critic(critic_id=entries.rows[0][0], critic_name=entries.rows[0][1])
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        return Critic.badCritic()


def deleteCritic(critic_id: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Critic WHERE CriticID={id}").format(
        id=sql.Literal(critic_id))
    return delete_from_table_macro(query)


# -------------functions for actor-----------------#

def addActor(actor: Actor) -> ReturnValue:
    query = sql.SQL("INSERT INTO Actor(ActorID, Name, Age, Height) VALUES({id},{name},{age},{height})").format(
        id=sql.Literal(actor.getActorID()), name=sql.Literal(actor.getActorName()),
        age=sql.Literal(actor.getAge()), height=sql.Literal(actor.getHeight()))
    return add_to_table_macro(query)

def getActorProfile(actor_id: int) -> Actor:
    query = sql.SQL("SELECT ActorID,Name,Age,Height FROM Actor WHERE ActorID={id}").format(
        id=sql.Literal(actor_id))
    entries = get_from_table_macro(query)
    try:
        return Actor(actor_id=entries.rows[0][0], actor_name=entries.rows[0][1], age=entries.rows[0][2], height=entries.rows[0][3])
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        return Actor.badActor()


def deleteActor(actor_id: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Actor WHERE ActorID={id}").format(
        id=sql.Literal(actor_id))
    return delete_from_table_macro(query)


# -------------functions for movie-----------------#

def addMovie(movie: Movie) -> ReturnValue:
    query = sql.SQL("INSERT INTO Movie(MovieName,Year,Genre) VALUES({mName}, {mYear}, {mGenre})").format(
        mName=sql.Literal(movie.getMovieName()), mYear=sql.Literal(movie.getYear()),
        mGenre=sql.Literal(movie.getGenre()))
    return add_to_table_macro(query)


def getMovieProfile(movie_name: str, year: int) -> Movie:
    query = sql.SQL("SELECT MovieName,Year,Genre FROM Movie WHERE MovieName = {mName} AND Year = {mYear}").format(
        mName=sql.Literal(movie_name), mYear=sql.Literal(year))
    entries = get_from_table_macro(query)
    try:
        return Movie(movie_name=entries.rows[0][0], year=entries.rows[0][1], genre=entries.rows[0][2])
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        return Movie.badMovie()


def deleteMovie(movie_name: str, year: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Movie WHERE MovieName={mName} AND Year={mYear}").format(
        mName=sql.Literal(movie_name), mYear=sql.Literal(year))
    return delete_from_table_macro(query)


# -------------functions for studio-----------------#

def addStudio(studio: Studio) -> ReturnValue:
    query = sql.SQL("INSERT INTO Studio(StudioID,Name) VALUES({id}, {name})").format(
        id=sql.Literal(studio.getStudioID()), name=sql.Literal(studio.getStudioName()))
    return add_to_table_macro(query)


def deleteStudio(studio_id: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Studio WHERE StudioID={id}").format(
        id=sql.Literal(studio_id))
    return delete_from_table_macro(query)


def getStudioProfile(studio_id: int) -> Studio:
    query = sql.SQL("SELECT StudioID,Name FROM Studio WHERE StudioID = {id}").format(id=sql.Literal(studio_id))
    entries = get_from_table_macro(query)
    try:
        return Studio(studio_id=entries.rows[0][0], studio_name=entries.rows[0][1])
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        return Studio.badStudio()


# -------------macro_api-----------------#

def add_to_table_macro_api(query):
    """gets a sql literal Add query and executes it, returns Return value, it's for API"""
    conn = None
    try:
        conn = Connector.DBConnector()
        rows_effected, _ = conn.execute(query)
        return ReturnValue.OK
    except DatabaseException.CHECK_VIOLATION as e:
        return ReturnValue.BAD_PARAMS
    except (DatabaseException.NOT_NULL_VIOLATION, DatabaseException.FOREIGN_KEY_VIOLATION) as e:
        return ReturnValue.NOT_EXISTS
    except DatabaseException.UNIQUE_VIOLATION as e:
        return ReturnValue.ALREADY_EXISTS
    except (DatabaseException.ConnectionInvalid, DatabaseException.UNKNOWN_ERROR, Exception) as e:
        return ReturnValue.ERROR
    finally:
        conn.close()
    pass


def delete_from_table_macro_api(query):
    """gets a sql literal del query and executes it, returns ReturnValue API version """
    conn = None
    try:
        conn = Connector.DBConnector()
        rows_effected, _ = conn.execute(query)
        if rows_effected != 1:
            raise DatabaseException.FOREIGN_KEY_VIOLATION("FOREIGN_KEY_VIOLATION")
        return ReturnValue.OK
    except (DatabaseException.NOT_NULL_VIOLATION, DatabaseException.FOREIGN_KEY_VIOLATION) as e:
        return ReturnValue.NOT_EXISTS
    except (DatabaseException.ConnectionInvalid, DatabaseException.UNKNOWN_ERROR, DatabaseException.UNIQUE_VIOLATION,
            DatabaseException.CHECK_VIOLATION, Exception) as e:
        return ReturnValue.ERROR
    finally:
        conn.close()
    pass


# -------------functions for critic rates-----------------#

def criticRatedMovie(movieName: str, movieYear: int, criticID: int, rating: int) -> ReturnValue:
    query = sql.SQL("INSERT INTO Rates(MovieName,Year,CriticID,Rating) VALUES({mName}, {mYear}, {cID},{r})").format(
        mName=sql.Literal(movieName), mYear=sql.Literal(movieYear), cID=sql.Literal(criticID), r=sql.Literal(rating))
    return add_to_table_macro_api(query)


def criticDidntRateMovie(movieName: str, movieYear: int, criticID: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Rates WHERE MovieName={mName} AND Year={mYear} AND CriticID={cID}").format(
        mName=sql.Literal(movieName), mYear=sql.Literal(movieYear), cID=sql.Literal(criticID))
    return delete_from_table_macro_api(query)


def debug_print_Rates():
    conn = Connector.DBConnector()
    query = sql.SQL("SELECT * FROM Rates")
    rows_effected, _ = conn.execute(query, True)
    pass


# -------------functions for actor playes-----------------#

def actorPlayedInMovie(movieName: str, movieYear: int, actorID: int, salary: int, roles: List[str]) -> ReturnValue:
    query = sql.SQL("INSERT INTO Plays(MovieName,Year,ActorID,Salary,Roles) VALUES({mName}, {mYear}, {aID},{sal}"
                    ",{roles})").format(
        mName=sql.Literal(movieName), mYear=sql.Literal(movieYear), aID=sql.Literal(actorID), sal=sql.Literal(salary),
        roles=sql.Literal(roles))
    return add_to_table_macro_api(query)


def actorDidntPlayInMovie(movieName: str, movieYear: int, actorID: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Plays WHERE MovieName={mName} AND Year={mYear} AND ActorID={aID}").format(
        mName=sql.Literal(movieName), mYear=sql.Literal(movieYear), aID=sql.Literal(actorID))
    return delete_from_table_macro_api(query)


def getActorsRoleInMovie(actorID: int, movieName: str, movieYear: int) -> List[str]:
    query = sql.SQL("SELECT * \
                    FROM UNNEST( \
                            (SELECT Roles \
                            FROM Plays \
                            WHERE ActorID={aID} AND MovieName={mName} AND Year={mYear}) \
                        ) AS roles \
                    ORDER BY roles").format(aID=sql.Literal(actorID), mName=sql.Literal(movieName), mYear=sql.Literal(movieYear))
    entries = get_from_table_macro(query)
    if entries == ReturnValue.ERROR:
        return []
    return list(*zip(*entries.rows))
def debug_print_Plays():
    conn = Connector.DBConnector()
    query = sql.SQL("SELECT * FROM Plays")
    rows_effected, _ = conn.execute(query, True)
    pass


# -------------functions for actor playes-----------------#

def studioProducedMovie(studioID: int, movieName: str, movieYear: int, budget: int, revenue: int) -> ReturnValue:
    query = sql.SQL(
        "INSERT INTO Produces(StudioID,MovieName,Year,Budget,Revenue) VALUES({sID},{mName},{mYear},{b},{r})").format(
        sID=sql.Literal(studioID), mName=sql.Literal(movieName), mYear=sql.Literal(movieYear),
        b=sql.Literal(budget), r=sql.Literal(revenue))
    return add_to_table_macro_api(query)


def studioDidntProduceMovie(studioID: int, movieName: str, movieYear: int) -> ReturnValue:
    query = sql.SQL("DELETE FROM Produces WHERE StudioID={sID} AND MovieName={mName} AND Year={mYear}").format(
        sID=sql.Literal(studioID), mName=sql.Literal(movieName), mYear=sql.Literal(movieYear))
    return delete_from_table_macro_api(query)


def debug_print_Produces():
    conn = Connector.DBConnector()
    query = sql.SQL("SELECT * FROM Produces")
    rows_effected, _ = conn.execute(query, True)
    pass


# ---------------------------------- BASIC API: ----------------------------------#

def averageRating(movieName: str, movieYear: int) -> float:
    query = sql.SQL("SELECT AVG(Rating) FROM Rates WHERE MovieName={mName} AND Year={mYear}").format(
        mName=sql.Literal(movieName), mYear=sql.Literal(movieYear))
    conn = Connector.DBConnector()
    try:
        rows_effected, entries = conn.execute(query)
        return float(entries.rows[0][0])
    except:
        return float(0.0)
    finally:
        conn.close()

#---------need to check wht if movie does not exist the table inclddes None---------#
#---------line 262 in the test---------------------#
def averageActorRating(actorID: int) -> float:
    conn = None
    query = sql.SQL("SELECT AVG(avg_rating) as avg_rating \
                    FROM ( \
                        SELECT COALESCE(AVG(r.Rating), 0) as avg_rating  \
                        FROM Plays p \
                        JOIN Movie m ON p.ActorID = {aID} AND m.MovieName = p.MovieName AND m.Year = p.Year\
                        LEFT JOIN Rates r ON r.MovieName = m.MovieName AND r.Year = m.Year \
                        GROUP BY m.MovieName, m.Year \
                    ) AS averageActorRating").format(aID=sql.Literal(actorID)) 
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        if(entries.rows and entries.rows[0][0] is not None):
            return entries.rows[0][0]
        else:
            return float(0.0)
    except Exception as e:
        print(e) #check if realy need to print exception!!!!
        return float(0.0)
    finally:
        conn.close()
    pass


def bestPerformance(actor_id: int) -> Movie:
    conn = None
    bm = Movie.badMovie()
    query = sql.SQL("SELECT \
                CASE \
                    WHEN p.ActorID IS NULL THEN {BMname} \
                    ELSE p.MovieName \
                END AS MovieName, \
                CASE \
                    WHEN p.ActorID IS NULL THEN {BMyear} \
                    ELSE p.Year \
                END AS Year, \
                COALESCE(AVG(r.Rating), 0) AS avg_rating \
                FROM Plays p \
                JOIN Movie m ON p.ActorID = {aID} AND m.MovieName = p.MovieName AND m.Year = p.Year \
                LEFT JOIN Rates r ON r.MovieName = m.MovieName AND r.Year = m.Year \
                GROUP BY p.MovieName, p.Year, p.ActorID\
                ORDER BY avg_rating DESC, p.Year ASC, p.MovieName DESC \
                LIMIT 1").format(aID=sql.Literal(actor_id),
                                 BMname=sql.Literal(bm.getMovieName()), BMyear=sql.Literal(bm.getYear()))
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return getMovieProfile(entries.rows[0][0], entries.rows[0][1])
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        #print(e)
        return Movie.badMovie()
    finally:
        conn.close()
    pass


def stageCrewBudget(movieName: str, movieYear: int) -> int:
    conn = None
    query = sql.SQL("SELECT \
                        CASE \
                            WHEN m.MovieName IS NOT NULL THEN COALESCE(a.Budget, 0) - COALESCE(SUM(p.Salary), 0) \
                            ELSE -1 \
                        END AS budget_difference \
                        FROM (SELECT * FROM Movie WHERE MovieName = {mName} AND Year = {mYear}) m \
                        LEFT JOIN Produces a ON a.MovieName = m.MovieName AND a.Year = m.Year \
                        LEFT JOIN Plays p ON p.MovieName = m.MovieName AND p.Year = m.Year \
                        GROUP BY m.MovieName, m.Year, a.Budget").format(mName=sql.Literal(movieName), mYear=sql.Literal(movieYear))
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        if (entries.rows):
            return entries.rows[0][0]
        else: #case that the movie is not in the movie table
            return -1
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        conn.close()
    pass


def overlyInvestedInMovie(movie_name: str, movie_year: int, actor_id: int) -> bool:
    conn = None
    # need to update according to views
    drop_view = sql.SQL("DROP VIEW IF EXISTS overlyInvestedInMovie")
    view = "CREATE VIEW overlyInvestedInMovie AS \
                        SELECT ActorID, (SELECT COUNT(*) FROM UNNEST(Roles) as role) as role_count\
                        FROM Plays\
                        WHERE MovieName = {mName} AND Year = {mYear}"   
    check = "SELECT \
                CASE \
                    WHEN (SELECT role_count FROM overlyInvestedInMovie WHERE ActorID = {aID}) >= \
                        ((SELECT SUM(role_count) FROM overlyInvestedInMovie ) / 2) THEN True \
                    ELSE False \
                END AS result \
            FROM overlyInvestedInMovie \
            WHERE ActorID = {aID}"
    query = sql.SQL(view + "; " + check).format(mName=sql.Literal(movie_name), mYear=sql.Literal(movie_year), aID=sql.Literal(actor_id))
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        if (entries.rows):
            return entries.rows[0][0]
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        if conn is not None:
            conn.execute(drop_view)
        conn.close()
    pass


# ---------------------------------- ADVANCED API: ----------------------------------


def franchiseRevenue() -> List[Tuple[str, int]]:
    conn = None
    query = sql.SQL("SELECT m.MovieName AS movie_name, COALESCE(SUM(p.Revenue), 0) AS total_revenue \
                    FROM Movie m \
                    LEFT JOIN Produces p ON m.MovieName = p.MovieName AND m.Year = p.Year\
                    GROUP BY m.MovieName \
                    ORDER BY m.MovieName DESC")        
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return entries.rows
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        conn.close()
    pass


def studioRevenueByYear() -> List[Tuple[str, int]]:
    conn = None
    query = sql.SQL("SELECT StudioID, Year, SUM(Revenue) AS total_revenue \
                    FROM Produces \
                    GROUP BY StudioID, Year \
                    ORDER BY StudioID DESC, Year DESC")        
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return entries.rows
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        conn.close()
    pass


def getFanCritics() -> List[Tuple[int, int]]:
    conn = None
    query = sql.SQL("SELECT r.CriticID, p.StudioID \
                    FROM Rates r\
                    JOIN Produces p\
                    ON r.MovieName = p.MovieName AND r.Year = p.Year \
                    GROUP BY r.CriticID, p.StudioID \
                    HAVING COUNT(*) = (SELECT COUNT(*) FROM Produces WHERE StudioID = p.StudioID) \
                    ORDER BY r.CriticID DESC, p.StudioID DESC")                        
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return entries.rows
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        conn.close()
    pass


def averageAgeByGenre() -> List[Tuple[str, float]]:
    conn = None
    query = sql.SQL("SELECT g.Genre, AVG(a.Age) AS average_age\
                    FROM (SELECT DISTINCT p.ActorID, m.Genre\
                    FROM Movie m\
                    JOIN Plays p ON m.MovieName = p.MovieName AND m.Year = p.Year) g\
                    LEFT JOIN actor a ON g.ActorID = a.ActorID\
                    GROUP BY g.Genre\
                    ORDER BY g.Genre")                                       
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return entries.rows
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        conn.close()
    pass


def getExclusiveActors() -> List[Tuple[int, int]]:
    conn = None
    query = sql.SQL("SELECT a.ActorID, p.StudioID \
                        FROM Plays a \
                        JOIN Produces p \
                        ON a.MovieName = p.MovieName AND a.Year = p.Year \
                        WHERE NOT EXISTS ( \
                            SELECT 1 b \
                            FROM Plays b \
                            LEFT JOIN Produces c \
                            ON b.MovieName = c.MovieName AND b.Year = c.Year \
                            WHERE b.ActorID = a.ActorID AND c.StudioID IS NULL \
                        ) AND NOT EXISTS ( \
                            SELECT 1 \
                            FROM Plays b \
                            JOIN Produces c \
                            ON b.MovieName = c.MovieName AND b.Year = c.Year \
                            WHERE b.ActorID = a.ActorID AND c.StudioID != p.StudioID \
                        ) \
                        GROUP BY a.ActorID, p.StudioID \
                        ORDER BY a.ActorID DESC")                                        
    try:
        conn = Connector.DBConnector()
        rows_effected, entries = conn.execute(query)
        return entries.rows
    # because id is unique so entries is either empty or has one row or entries is ReturnValue,
    # if it has one row it's ok
    # if it is empty or is a ReturnValue we will get an exception which is also ok
    except Exception as e:
        print(e)
    finally:
        conn.close()
    pass


