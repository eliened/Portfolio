import unittest
from decimal import Decimal

from Solution import *

GENRES = ['Drama', 'Action', 'Comedy', 'Horror']


class MovieStatsTest(unittest.TestCase):
    def setUp(self) -> None:
        dropTables()
        createTables()
        self._fillN(n=2)
        clearTables()
        self._fillN(n=9)

    def _fillN(self, n: int) -> None:
        for j in range(1, n + 1):
            self.assertEqual(addMovie(Movie(f'm{j}', 1990 + j, GENRES[j % 4])), ReturnValue.OK)
            self.assertEqual(addActor(Actor(j, f'a{j}', 20 + 3 * j, j * 10)), ReturnValue.OK)
            self.assertEqual(addStudio(Studio(j, f's{j}')), ReturnValue.OK)
            self.assertEqual(addCritic(Critic(j, f'c{j}')), ReturnValue.OK)
            self.assertEqual(criticRatedMovie(f'm{j}', 1990 + j, j, j % 5 + 1), ReturnValue.OK)
            self.assertEqual(actorPlayedInMovie(f'm{j}', 1990 + j, j, j * 1000, [f'r{j}']), ReturnValue.OK)
            self.assertEqual(studioProducedMovie(j, f'm{j}', 1990 + j, j * 100, j * 700), ReturnValue.OK)

    def test_addCritic(self):
        self.assertEqual(addCritic(Critic(1, 'c1')), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(addCritic(Critic(0, 'c')), ReturnValue.BAD_PARAMS)
        self.assertEqual(addCritic(Critic(-5, 'c')), ReturnValue.BAD_PARAMS)

    def test_getCriticProfile(self):
        self.assertEqual(getCriticProfile(1), Critic(1, 'c1'))
        self.assertEqual(getCriticProfile(20), Critic.badCritic())
        self.assertEqual(getCriticProfile(0), Critic.badCritic())
        self.assertEqual(getCriticProfile(-5), Critic.badCritic())

    def test_deleteCritic(self):
        self.assertEqual(deleteCritic(1), ReturnValue.OK)
        self.assertEqual(deleteCritic(0), ReturnValue.NOT_EXISTS)
        self.assertEqual(deleteCritic(20), ReturnValue.NOT_EXISTS)

    def test_addMovie(self):
        self.assertEqual(addMovie(Movie('m1', 1991, GENRES[0])), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(addMovie(Movie('m1', 1991, GENRES[1])), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(addMovie(Movie('m1', 1992, GENRES[0])), ReturnValue.OK)
        self.assertEqual(addMovie(Movie('m', 1894, GENRES[1])), ReturnValue.BAD_PARAMS)
        self.assertEqual(addMovie(Movie('m', 2000, 'Spooky')), ReturnValue.BAD_PARAMS)

    def test_getMovieProfile(self):
        self.assertEqual(getMovieProfile('m1', 1991), Movie('m1', 1991, GENRES[1]))
        self.assertEqual(getMovieProfile('m1', 1990), Movie.badMovie())
        self.assertEqual(getMovieProfile('m1', 1), Movie.badMovie())
        self.assertEqual(getMovieProfile('m', 2000), Movie.badMovie())

    def test_deleteMovie(self):
        self.assertEqual(deleteMovie('m1', 1991), ReturnValue.OK)
        self.assertEqual(deleteMovie('m', 0), ReturnValue.NOT_EXISTS)

    def test_addActor(self):
        self.assertEqual(addActor(Actor(1, 'a1', 23, 10)), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(addActor(Actor(20, 'a1', 23, 10)), ReturnValue.OK)
        self.assertEqual(addActor(Actor(21, 'a1', 23, 0)), ReturnValue.BAD_PARAMS)
        self.assertEqual(addActor(Actor(21, 'a1', 0, 10)), ReturnValue.BAD_PARAMS)
        self.assertEqual(addActor(Actor(0, 'a1', 23, 10)), ReturnValue.BAD_PARAMS)
        self.assertEqual(addActor(Actor(21, 'a1', 23, -5)), ReturnValue.BAD_PARAMS)
        self.assertEqual(addActor(Actor(21, 'a1', -5, 10)), ReturnValue.BAD_PARAMS)
        self.assertEqual(addActor(Actor(-5, 'a1', 23, 10)), ReturnValue.BAD_PARAMS)

    def test_getActorProfile(self):
        self.assertEqual(getActorProfile(1), Actor(1, 'a1', 23, 10))
        self.assertEqual(getActorProfile(0), Actor.badActor())
        self.assertEqual(getActorProfile(20), Actor.badActor())
        self.assertEqual(getActorProfile(-5), Actor.badActor())

    def test_deleteActor(self):
        self.assertEqual(deleteActor(1), ReturnValue.OK)
        self.assertEqual(deleteActor(0), ReturnValue.NOT_EXISTS)
        self.assertEqual(deleteActor(-5), ReturnValue.NOT_EXISTS)
        self.assertEqual(deleteActor(20), ReturnValue.NOT_EXISTS)

    def test_addStudio(self):
        self.assertEqual(addStudio(Studio(1, 's1')), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(addStudio(Studio(20, 's1')), ReturnValue.OK)
        self.assertEqual(addStudio(Studio(0, 's')), ReturnValue.BAD_PARAMS)
        self.assertEqual(addStudio(Studio(-5, 's')), ReturnValue.BAD_PARAMS)

    def test_getStudioProfile(self):
        self.assertEqual(getStudioProfile(1), Studio(1, 's1'))
        self.assertEqual(getStudioProfile(20), Studio.badStudio())
        self.assertEqual(getStudioProfile(0), Studio.badStudio())
        self.assertEqual(getStudioProfile(-5), Studio.badStudio())

    def test_deleteStudio(self):
        self.assertEqual(deleteStudio(1), ReturnValue.OK)
        self.assertEqual(deleteStudio(0), ReturnValue.NOT_EXISTS)
        self.assertEqual(deleteStudio(20), ReturnValue.NOT_EXISTS)
        self.assertEqual(deleteStudio(-5), ReturnValue.NOT_EXISTS)

    def test_crticRatedMovie(self):
        self.assertEqual(criticRatedMovie('m1', 1991, 1, 2), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(criticRatedMovie('m1', 1990, 1, 2), ReturnValue.NOT_EXISTS)
        self.assertEqual(criticRatedMovie('m1', 1991, 20, 2), ReturnValue.NOT_EXISTS)
        self.assertEqual(criticRatedMovie('m1', 1991, 2, 2), ReturnValue.OK)
        self.assertEqual(criticRatedMovie('m1', 1991, 3, 6), ReturnValue.BAD_PARAMS)
        self.assertEqual(criticRatedMovie('m1', 1991, 3, 0), ReturnValue.BAD_PARAMS)

    def test_CriticDidntRateMovie(self):
        self.assertEqual(criticDidntRateMovie('m1', 1991, 1), ReturnValue.OK)
        self.assertEqual(criticRatedMovie('m1', 1991, 1, 2), ReturnValue.OK)
        self.assertEqual(criticDidntRateMovie('m1', 1991, 2), ReturnValue.NOT_EXISTS)
        self.assertEqual(criticDidntRateMovie('m2', 2000, 2), ReturnValue.NOT_EXISTS)
        self.assertEqual(criticDidntRateMovie('m2', 1992, 20), ReturnValue.NOT_EXISTS)

    def test_actorPlayedInMovie(self):
        self.assertEqual(actorPlayedInMovie('m1', 1991, 1, 1000, ['r1']), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 1, 10, ['r1']), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 1, 1000, ['r2']), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(actorPlayedInMovie('m2', 1991, 1, 1000, ['r2']), ReturnValue.NOT_EXISTS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 20, 1000, ['r2']), ReturnValue.NOT_EXISTS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 2, 0, ['r2']), ReturnValue.BAD_PARAMS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 2, -5, ['r2']), ReturnValue.BAD_PARAMS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 2, 1000, []), ReturnValue.BAD_PARAMS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 2, 1000, ['r', None, 'r2']), ReturnValue.BAD_PARAMS)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 2, 1000, [None]), ReturnValue.BAD_PARAMS)

    def test_actorDidntPlatInMovie(self):
        self.assertEqual(actorDidntPlayInMovie('m1', 1991, 1), ReturnValue.OK)
        self.assertEqual(actorPlayedInMovie('m1', 1991, 1, 1000, ['r1']), ReturnValue.OK)
        self.assertEqual(actorDidntPlayInMovie('m2', 1991, 1), ReturnValue.NOT_EXISTS)
        self.assertEqual(actorDidntPlayInMovie('m2', 1992, 1), ReturnValue.NOT_EXISTS)
        self.assertEqual(actorDidntPlayInMovie('m2', 1992, 20), ReturnValue.NOT_EXISTS)

    def test_getActorsRoleInMovie(self):
        self.assertEqual(getActorsRoleInMovie(1, 'm1', 1991), ['r1'])
        self.assertEqual(getActorsRoleInMovie(2, 'm1', 1991), [])
        self.assertEqual(getActorsRoleInMovie(20, 'm1', 1991), [])
        self.assertEqual(getActorsRoleInMovie(2, 'm', 1991), [])
        self.assertEqual(getActorsRoleInMovie(2, 'm1', 1), [])
        actorPlayedInMovie('m1', 1991, 2, 100, ['Aba', 'Baba', 'Caba'])
        self.assertEqual(getActorsRoleInMovie(2, 'm1', 1991), ['Caba', 'Baba', 'Aba'])

    def test_studioProducedMovie(self):
        self.assertEqual(studioProducedMovie(1, 'm1', 1991, 1100, 700), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(studioProducedMovie(1, 'm1', 1991, 1100, 500), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(studioProducedMovie(1, 'm1', 1991, 10, 700), ReturnValue.ALREADY_EXISTS)
        self.assertEqual(studioProducedMovie(2, 'm1', 1991, 1100, 700), ReturnValue.ALREADY_EXISTS)
        addMovie(Movie('m20', 2000, GENRES[0]))
        self.assertEqual(studioProducedMovie(20, 'm20', 2000, 100, 700), ReturnValue.NOT_EXISTS)
        self.assertEqual(studioProducedMovie(2, 'm20', 2010, 100, 700), ReturnValue.NOT_EXISTS)
        self.assertEqual(studioProducedMovie(2, 'm20', 2000, -1, 700), ReturnValue.BAD_PARAMS)
        self.assertEqual(studioProducedMovie(2, 'm20', 2000, 1000, -700), ReturnValue.BAD_PARAMS)
        self.assertEqual(studioProducedMovie(2, 'm20', 2000, 0, 0), ReturnValue.OK)

    def test_studioDidntProducedMovie(self):
        self.assertEqual(studioDidntProduceMovie(1, 'm1', 1991), ReturnValue.OK)
        self.assertEqual(studioProducedMovie(1, 'm1', 1991, 1100, 700), ReturnValue.OK)
        self.assertEqual(studioDidntProduceMovie(2, 'm1', 1991), ReturnValue.NOT_EXISTS)
        self.assertEqual(studioDidntProduceMovie(20, 'm1', 1991), ReturnValue.NOT_EXISTS)
        self.assertEqual(studioDidntProduceMovie(2, 'm', 1991), ReturnValue.NOT_EXISTS)

    def test_averageRating(self):
        self.assertEqual(averageRating('m20', 2000), 0)
        addMovie(Movie('m20', 2000, GENRES[0]))
        self.assertEqual(averageRating('m20', 2000), 0)
        criticRatedMovie('m20', 2000, 1, 1)
        self.assertEqual(averageRating('m20', 2000), 1)
        criticRatedMovie('m20', 2000, 2, 2)
        self.assertEqual(averageRating('m20', 2000), 1.5)
        criticRatedMovie('m20', 2000, 3, 3)
        self.assertEqual(averageRating('m20', 2000), 2)

    def test_averageActorRating(self):
        self.assertEqual(averageActorRating(20), 0)
        addActor(Actor(20, 'a20', 20, 200))
        self.assertEqual(averageActorRating(20), 0)
        for j in range(1, 5):
            addMovie(Movie(f'm2{j}', 2000 + j, GENRES[0]))
            actorPlayedInMovie(f'm2{j}', 2000 + j, 20, 100, ['r'])
        self.assertEqual(averageActorRating(20), 0)
        criticRatedMovie('m21', 2001, 1, 4)
        self.assertEqual(averageActorRating(20), 1)
        criticRatedMovie('m21', 2001, 2, 4)
        self.assertEqual(averageActorRating(20), 1)
        criticRatedMovie('m22', 2002, 1, 4)
        self.assertEqual(averageActorRating(20), 2)
        criticRatedMovie('m21', 2001, 3, 5)
        self.assertEqual(round(averageActorRating(20), 6), round(Decimal((((4 * 2 + 5) / 3) + 4) / 4), 6))

    def test_bestPerformance(self):
        self.assertEqual(bestPerformance(20), Movie.badMovie())
        addActor(Actor(20, 'a20', 20, 200))
        self.assertEqual(bestPerformance(20), Movie.badMovie())
        for j in range(1, 4):
            addMovie(Movie(f'm2{j}', 2000 + j, GENRES[0]))
            actorPlayedInMovie(f'm2{j}', 2000 + j, 20, 100, ['r'])
        criticRatedMovie('m21', 2001, 1, 1)
        self.assertEqual(bestPerformance(20), Movie('m21', 2001, GENRES[0]))
        criticRatedMovie('m22', 2002, 1, 1)
        self.assertEqual(bestPerformance(20), Movie('m21', 2001, GENRES[0]))
        criticRatedMovie('m22', 2002, 2, 3)
        self.assertEqual(bestPerformance(20), Movie('m22', 2002, GENRES[0]))
        addMovie(Movie('z', 2002, GENRES[0]))
        actorPlayedInMovie('z', 2002, 20, 10, ['r'])
        criticRatedMovie('z', 2002, 1, 2)
        self.assertEqual(bestPerformance(20), Movie('z', 2002, GENRES[0]))
        actorDidntPlayInMovie('z', 2002, 20)
        self.assertEqual(bestPerformance(20), Movie('m22', 2002, GENRES[0]))

    def test_stageCrewBudget(self):
        self.assertEqual(stageCrewBudget('m20', 2000), -1)
        addMovie(Movie('m20', 2000, GENRES[0]))
        self.assertEqual(stageCrewBudget('m20', 2000), 0)
        for j in range(1, 4):
            actorPlayedInMovie('m20', 2000, j, j * 10, ['r'])
        self.assertEqual(stageCrewBudget('m20', 2000), 0)
        studioProducedMovie(1, 'm20', 2000, 100, 100)
        self.assertEqual(stageCrewBudget('m20', 2000), 100 - (10 + 20 + 30))
        actorDidntPlayInMovie('m20', 2000, 1)
        self.assertEqual(stageCrewBudget('m20', 2000), 100 - (20 + 30))

    def test_overlyInvestedInMovie(self):
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 1), False)
        self.assertEqual(overlyInvestedInMovie('m1', 1991, 20), False)
        addMovie(Movie('m20', 2000, GENRES[0]))
        actorPlayedInMovie('m20', 2000, 1, 10, ['r1'])
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 1), True)
        actorPlayedInMovie('m20', 2000, 2, 10, ['r2'])
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 1), True)
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 2), True)
        actorPlayedInMovie('m20', 2000, 3, 10, ['r3', 'r4', 'r5'])
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 1), False)
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 2), False)
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 3), True)
        self.assertEqual(overlyInvestedInMovie('m20', 2000, 4), False)

    def test_frenchiseRevenue(self):
        result = [(f'm{i}', i * 700) for i in range(9, 0, -1)]
        self.assertEqual(franchiseRevenue(), result)
        addMovie(Movie('m1', 2000, GENRES[0]))
        studioProducedMovie(3, 'm1', 2000, 10, 3000)
        result[-1] = (result[-1][0], result[-1][1] + 3000)
        self.assertEqual(franchiseRevenue(), result)
        addMovie(Movie('m', 2000, GENRES[0]))
        result.append(('m', 0))
        self.assertEqual(franchiseRevenue(), result)

    def test_getFanCritics(self):
        result = [(i, i) for i in range(9, 0, -1)]
        self.assertEqual(getFanCritics(), result)
        criticRatedMovie('m2', 1992, 1, 1)
        result = result[:-1] + [(1, 2), (1, 1)]
        self.assertEqual(getFanCritics(), result)
        addMovie(Movie('m20', 2000, GENRES[0]))
        studioProducedMovie(1, 'm20', 2000, 10, 10)
        result = result[:-1]
        self.assertEqual(getFanCritics(), result)

    def test_averageAgeByGenre(self):
        result = [('Action', Decimal((sum([20 + 3 * i for i in [1, 5, 9]])) / 3)),
                  ('Comedy', Decimal((sum([20 + 3 * i for i in [2, 6]])) / 2)),
                  ('Drama', Decimal((sum([20 + 3 * i for i in [4, 8]])) / 2)),
                  ('Horror', Decimal((sum([20 + 3 * i for i in [3, 7]])) / 2))]
        self.assertEqual(averageAgeByGenre(), result)
        # ['Drama', 'Action', 'Comedy', 'Horror']
        for i, genre in zip(range(1, 8), GENRES * 2):
            addMovie(Movie(f'movie{i}', 1900 + i, genre))
        for i in range(10, 16):
            addActor(Actor(i, f'actor{i}', i, 1))
        actorPlayedInMovie('movie1', 1901, 10, 1, ['r'])  # Drama, 10
        result[2] = ('Drama', round(((result[2][1] * 2) + 10) / 3, 6))
        self.assertEqual([(g, round(a, 6)) for (g, a) in averageAgeByGenre()], result)
        actorPlayedInMovie('movie5', 1905, 10, 1, ['r'])  # should have no effect
        self.assertEqual([(g, round(a, 6)) for (g, a) in averageAgeByGenre()], result)

    def test_getExclusiveActors(self):
        result = [(i, i) for i in range(9, 0, -1)]
        self.assertEqual(getExclusiveActors(), result)
        actorPlayedInMovie('m2', 1992, 1, 1, ['r'])
        result = result[:-1]
        self.assertEqual(getExclusiveActors(), result)
        addMovie(Movie('m20', 2000, GENRES[0]))
        studioProducedMovie(2, 'm20', 2000, 1, 1)
        self.assertEqual(getExclusiveActors(), result)
        actorPlayedInMovie('m20', 2000, 2, 1, ['r'])
        print('res_test', getExclusiveActors())
        self.assertEqual(getExclusiveActors(), result)
        actorDidntPlayInMovie('m20', 2000, 2)
        actorDidntPlayInMovie('m2', 1992, 2)
        result = result[:-1]
        self.assertEqual(getExclusiveActors(), result)




if __name__ == '__main__':
    unittest.main()
