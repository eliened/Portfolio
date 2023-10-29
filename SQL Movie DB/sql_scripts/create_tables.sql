--Creation of critics

CREATE TABLE IF NOT EXISTS Critics(cId INT PRIMARY KEY, name VARCHAR(100) NOT NULL);

--Creation of movies

CREATE TABLE Movies (
    name VARCHAR(100),
    year INT,
    genre VARCHAR(6) NOT NULL,
    PRIMARY KEY (name,year)
);

--Creation of actors

CREATE TABLE Actors (aId INT PRIMARY KEY,name VARCHAR(100) NOT NULL ,age INT NOT NULL CHECK ( age > 0 ),height INT NOT NULL CHECK ( height > 0 ));

--Creation of Studios

CREATE TABLE Studio (
    sId INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

--Creation of MovieCritics

CREATE TABLE MovieCritics (
    movie_name VARCHAR(100) ,
    movie_year INT,
    cId INT,
    rating INT NOT NULL CHECK ( rating >= 1 and rating <= 5 ),
    FOREIGN KEY (movie_name, movie_year) REFERENCES Movies(name, year),
    FOREIGN KEY (cId) REFERENCES Critics(cId)
);

--Creation of MovieActors

CREATE TABLE MovieActors (
    movie_name VARCHAR(100),
    movie_year INT,
    aId INT,
    salary INT NOT NULL CHECK ( salary > 0 ) ,
    role VARCHAR(100) NOT NULL ,
    FOREIGN KEY (movie_name, movie_year) REFERENCES Movies(name, year),
    FOREIGN KEY (aId) REFERENCES Actors(aId)
);

--Creation of Productions

CREATE TABLE Production (
    movie_name VARCHAR(100),
    movie_year INT,
    sId INT,
    budget INT,
    revenue INT,
    FOREIGN KEY (movie_name, movie_year) REFERENCES Movies(name, year),
    FOREIGN KEY (sId) REFERENCES Studio(sId)
);


