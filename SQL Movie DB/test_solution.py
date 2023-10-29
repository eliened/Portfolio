import unittest
from Solution import *


def f():
    print("caca")

class MyTestCase(unittest.TestCase):


    def test_addActor(self):
        # TEST addActor
        for i in range(1,7):
            self.assertEqual(addActor(Actor(i, f'actor_{i}',i * 10, i * 10)),ReturnValue.OK)

        self.assertEqual(addActor(Actor(1, f'actor_{1}',10, 10)),ReturnValue.ALREADY_EXISTS)
        self.assertEqual(addActor(Actor(10, f'actor_{10}',0, 10)),ReturnValue.BAD_PARAMS)
        self.assertEqual(addActor(Actor(10, f'actor_{10}',10, -5)),ReturnValue.BAD_PARAMS)
        print("AddActor OK")









if __name__ == '__main__':


    unittest.main()


