--Vue qui contient chaque film avec sa note moyenne
CREATE VIEW AverageRating AS
SELECT genre, movie_name,movie_year, avg(rating) as avg_rating
FROM moviecritics mc
JOIN movies m on mc.movie_name = m.name and mc.movie_year = m.year
group by movie_name, movie_year, genre;


--Vue qui contient Les acteurs sans les roles (Pour eviter qu'ils soient compté plusieurs fois dans certaines
--queries)
CREATE VIEW ActorsMoviesNoRole AS
    SELECT movie_name, movie_year,aid,salary
    FROM movieactors
    GROUP BY movie_name, movie_year,aid,salary;

--La note moyenne de chaque film auquel un acteur a participé
CREATE VIEW avgRatingActors AS
    SELECT ma.aid, ma.movie_name, ma.movie_year, avg_rating, genre
    FROM averagerating ag
    JOIN ActorsMoviesNoRole ma ON ag.movie_year = ma.movie_year AND ag.movie_name = ma.movie_name
    UNION
    SELECT ma.aid, ma.movie_name, ma.movie_year, 0, genre
    FROM averagerating ag
    RIGHT JOIN ActorsMoviesNoRole ma ON ag.movie_year = ma.movie_year AND ag.movie_name = ma.movie_name
    WHERE ag.avg_rating IS NULL;




--Average rating
SELECT avg_rating
FROM averageRating
WHERE movie_name = 'SpiderMan' and movie_year = 2000;

--AverageActorRating
--La moyenne des notes des films ou l'acteur donné a joué.
SELECT avg(avg_rating)
FROM avgRatingActors
WHERE aid = 1;

--bestPerformance
--La le films ayant la meilleure note dans le quel l'acteur donné a joué
SELECT movie_name,movie_year, genre
from avgRatingActors
where aid = 999 and avg_rating >= ALL(
    SELECT avg_rating FROM avgRatingActors where aid = 999
    )
ORDER BY movie_year, movie_name DESC limit 1;


--crew budget
SELECT  budget - sum(salary) as crew_budeget
from productions p
join ActorsMoviesNoRole am on p.movie_name = am.movie_name and p.movie_year = am.movie_year
WHERE p.movie_name = '999' and p.movie_year = 999
group by (p.movie_year, p.movie_name, p.sid, p.budget);



--overly invested in movie
SELECT (n_roles_actors.cnt / n_roles_movie.cnt) > 0.5
FROM (SELECT aid, movie_name, movie_year, count(role_played) as cnt
        FROM movieactors
        GROUP BY aid, movie_name, movie_year) n_roles_actors
JOIN (SELECT movie_name, movie_year, count(role_played) as cnt
        FROM movieactors
        GROUP BY movie_name, movie_year) n_roles_movie
On n_roles_movie.movie_name = n_roles_actors.movie_name and n_roles_movie.movie_year = n_roles_actors.movie_year
Where n_roles_actors.movie_year = 2000 and n_roles_actors.movie_name = 'SpiderMan2' and aid = 1;


--Movie Revenue

(SELECT m.name as m_name, sum(revenue) as movie_revenue
FROM movies m
JOIN productions p on p.movie_name = m.name and p.movie_year = m.year
GROUP BY m.name)
UNION --Add the movies that weren't produced as 0 revenue
(SELECT m.name as m_name, 0 as movie_revenue
FROM movies m
LEFT JOIN productions p on p.movie_name = m.name and p.movie_year = m.year
WHERE p.revenue IS NULL
GROUP BY m.name)
order by m_name desc
;

--Studio Revenue
SELECT sid, movie_year, sum(revenue)
FROM productions
group by sid, movie_year
order by sid desc, movie_year desc;


-- AverageAgeByGenre
SELECT genre, AVG(age)
FROM
    (SELECT distinct a.aid, genre, age
    FROM actorsmoviesnorole as am
    JOIN actors a on am.aid = a.aid
    JOIN movies m on movie_year = m.year and movie_name = m.name) B
group by genre;



--Exclusive Actor
CREATE VIEW actorStudio AS
SELECT aid, sid
FROM ActorsMoviesNoRole am
JOIN productions p on p.movie_name = am.movie_name and p.movie_year = am.movie_year;


SELECT aid, sid
FROM actorStudio
WHERE aid IN
    (SELECT aid
    FROM actorStudio
    GROUP BY aid
    having count(distinct sid) = 1);

select cid, A.sid
FROM
    (SELECT sid, count(*) as prod_cnt
    FROM productions p
    GROUP BY sid) A
JOIN
    (SELECT cid, sid, count(*) as critic_on_studio_cnt
    FROM moviecritics mc
    JOIN productions p on mc.movie_year = p.movie_year and mc.movie_name = p.movie_name
    GROUP BY cid, sid) B ON A.sid = B.sid
WHERE A.prod_cnt = B.critic_on_studio_cnt

--stage crew budget
SELECT name, year, budget - sum_salaries as stage_budget
FROM movieprodactors
WHERE budget IS NOT NULL and sum_salaries IS NOT NULL
UNION
SELECT name, year, 0 as stage_budget
FROM movieprodactors
WHERE budget IS NULL
UNION
SELECT name, year, budget  as stage_budget
FROM movieprodactors
WHERE sum_salaries IS NULL and budget IS NOT NULL;



CREATE VIEW movieprodactors AS
    select m.name, m.year, budget, sum(salary) as sum_salaries
    from movies m
    LEFT OUTER join productions p on m.name = p.movie_name and m.year = p.movie_year
    LEFT OUTER join actorsmoviesnorole a on p.movie_name = a.movie_name and p.movie_year = a.movie_year
    group by m.name, m.year, budget;






SELECT m.name as m_name, sum(revenue) as movie_revenue
FROM movies m
JOIN productions p on p.movie_name = m.name and p.movie_year = m.year
GROUP BY m.name;

SELECT m.name as m_name, 0 as movie_revenue
FROM movies m
LEFT JOIN productions p on p.movie_name = m.name and p.movie_year = m.year
GROUP BY m.name
having sum(p.revenue) IS NULL;