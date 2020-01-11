# -*- coding: utf-8 -*-

"""
This file will test the rossum.py-file
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

from biosim.rossum import Island, Savannah, Herbivores
import pytest
import numpy as np


class TestIsland:
    def test_island_call(self):
        """Default constructor callable"""
        a = Island()
        assert isinstance(a, Island)

    def test_perimeter_error(self):
        """Testing if ValueError is raised when the perimeter contains something other than ocean"""
        a = "OOO\nOJJ\nOOO"
        aa = Island(a)
        with pytest.raises(ValueError):
            aa.limit_map_vals()

    def test_unvalid_terrain_error(self):
        """Tests if the map input contains unvalid terrain-types"""
        a = "OOO\nOWO\nOOO"
        aa = Island(a)
        with pytest.raises(ValueError):
            aa.limit_map_vals()

    def test_unvaild_type(self):
        """Tests if the code raises TypeError if the map input is not a string"""
        a = [['O', 'O', 'O'], ['O', 'J', 'O'], ['O', 'O', 'O']]
        with pytest.raises(TypeError):
            Island(a)

    def test_output_np_array(self):
        """Tests if the output map is right, when input has no error"""
        a = "OOO\nOJO\nOOO"
        aa = Island(a)
        el_map = [['O', 'O', 'O'], ['O', 'J', 'O'], ['O', 'O', 'O']]
        assert np.array_equal(aa.fetch_map(), el_map)

    def test_return_neturtype(self):
        a = "OOO\nOJO\nOOO"
        aa = Island(a)
        pos = (0, 0)
        assert aa.fetch_naturetype(pos) == 'O'


class TestSavannah:
    def test_savannah_call(self):
        """Default constructor callable"""
        a = Savannah()
        assert isinstance(a, Savannah)

    def test_default_values(self):
        a = Savannah()
        assert a.fsav_max == 300
        assert a.alpha == 0.3

    def test_values(self):
        a = Savannah(fsav_max=200, alpha=0.4)
        assert a.fsav_max == 200
        assert a.alpha == 0.4

    def test_set_food(self):
        a = Savannah(fsav_max=300)
        a.set_food((1, 3))
        assert a.food == {(1, 3): 300}

    def test_set_more_food(self):
        a = Savannah(fsav_max=300)
        for i in range(3):
            for j in range(3):
                a.set_food((i, j))
        assert a.food[(0, 2)] == 300
        assert a.food[(1, 1)] == 300

    def test_grow_food(self):
        a = Savannah(fsav_max=300, alpha=0.3)
        a.set_food((1, 1))
        a.grow_food((1, 1))
        assert a.food[1, 1] == (300 + 0.3 * (300 - 300))

    def test_multiple_grow_food(self):
        a = Savannah()
        for i in range(3):
            for j in range(3):
                a.set_food((i, j))
        for i in range(3):
            for j in range(3):
                a.grow_food((i, j))
        assert a.food[0, 0] == (300 + 0.3 * (300 - 300))
        assert a.food[1, 1] == (300 + 0.3 * (300 - 300))
        assert a.food[2, 2] == (300 + 0.3 * (300 - 300))
        assert a.food[1, 2] == (300 + 0.3 * (300 - 300))

    def test_food_gets_eaten_with_less_food(self):
        a = Savannah(fsav_max=0)
        b = Savannah(fsav_max=5)
        c = Savannah()
        a.set_food((1, 1))
        b.set_food((1, 2))
        c.set_food((2, 2))
        aa = a.food_gets_eaten((1, 1))
        bb = b.food_gets_eaten((1, 2))
        cc = c.food_gets_eaten((2, 2))
        assert aa == 0
        assert a.food[(1, 1)] == 0
        assert bb == 5
        assert b.food[(1, 2)] == 0
        assert cc == 10
        assert c.food[(2, 2)] == 290


class TestHerbivores:
    def test_herbivores_call(self):
        """Default constructor callable"""
        a = Herbivores()
        assert isinstance(a, Herbivores)

    def test_default_values(self):
        a = Herbivores()
        assert a.w_birth == 8.0
        assert a.sigma_birth == 8.0
        assert a.beta == 1.5
        assert a.eta == 0.05
        assert a.a_half == 40.0
        assert a.phi_age == 0.2
        assert a.w_half == 10.0
        assert a.phi_weight == 0.1
        assert a.mu == 0.25
        assert a.lambda1 == 1.0
        assert a.gamma == 0.2
        assert a.zeta == 3.5
        assert a.xi == 1.2
        assert a.omega == 0.4

    def test_set_values(self):
        a = Herbivores(w_birth=34, sigma_birth=12)
        assert a.w_birth == 34
        assert a.sigma_birth == 12

    def test_add_animals_simple(self):
        added_dict = [{'loc': (3, 1), 'pop': [{'species': 'Herbievore', 'age': 0.1, 'Weight': 1.3}]}]
        a = Herbivores()
        a.add_animal(added_dict)
        assert a.herbs == {(3, 1): [{'species': 'Herbievore', 'age': 0.1, 'Weight': 1.3}]}

    def test_add_animals_twice(self):
        added_dict = [{'loc': (3, 1), 'pop': [{'species': 'Herbievore', 'age': 0.1, 'Weight': 1.3}]}]
        a = Herbivores()
        a.add_animal(added_dict)
        a.add_animal(added_dict)
        assert a.herbs == {(3, 1): [{'species': 'Herbievore', 'age': 0.1, 'Weight': 1.3},
                                    {'species': 'Herbievore', 'age': 0.1, 'Weight': 1.3}]}

    def test_add_multiple_animals(self):
        added_dict = [{'loc': (3, 1), 'pop': [{'species': 'Herbievore', 'age': 16, 'Weight': 1356.3},
                                    {'species': 'Herbievore', 'age': 113, 'Weight': 1323}]}]
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Herbievore', 'age': 12, 'Weight': 21.3},
                                    {'species': 'Herbievore', 'age': 123, 'Weight': 321.3}]},
                      {'loc': (3, 1), 'pop': [{'species': 'Herbievore', 'age': 166, 'Weight': 135.3},
                                              {'species': 'Herbievore', 'age': 11, 'Weight': 323}]}]
        a = Herbivores()
        a.add_animal(added_dict)
        a.add_animal(added_list)
        assert a.herbs[(3, 1)] == [{'species': 'Herbievore', 'age': 16, 'Weight': 1356.3},
                                   {'species': 'Herbievore', 'age': 113, 'Weight': 1323},
                                   {'species': 'Herbievore', 'age': 166, 'Weight': 135.3},
                                   {'species': 'Herbievore', 'age': 11, 'Weight': 323}]
        assert a.herbs[(3, 3)] == [{'species': 'Herbievore', 'age': 12, 'Weight': 21.3},
                                    {'species': 'Herbievore', 'age': 123, 'Weight': 321.3}]

    def test_calculate_fitness(self):
        added_list = [{'loc': (3, 1), 'pop': [{'species': 'Herbievore', 'age': 1, 'weight': 1.3}]}]
        a = Herbivores()
        a.add_animal(added_list)
        a.add_animal(added_list)
        a.add_animal(added_list)
        a.calculate_fitness((3,1))
        assert abs(a.herbs[(3, 1)][1]['fitness'] - 0.7044570573) < 0.001