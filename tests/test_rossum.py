# -*- coding: utf-8 -*-

"""
This file will test the island.py-file
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

from biosim.island import Island
from biosim.animal import Herbivores, Carnivores, Animal
import pytest
import numpy as np
import random
np.random.seed(seed=1)
random.seed(1)


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

    def test_return_naturetype(self):
        a = "OOO\nOJO\nOOO"
        aa = Island(a)
        pos = (0, 0)
        assert aa.fetch_naturetype(pos) == 'O'

    def test_default_values(self):
        a = Island()
        assert a.fsav_max == 300
        assert a.alpha == 0.3

    def test_values(self):
        a = Island(fsav_max=200, alpha=0.4)
        assert a.fsav_max == 200
        assert a.alpha == 0.4

    @pytest.fixture()
    def simple_island(self):
        return Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")

    @pytest.fixture()
    def savannah_island(self):
        return Island("OOOOO\nOSSSO\nOSSSO\nOSSSO\nOOOOO")

    def test_set_params(self):
        a = self.simple_island()
        a.set_new_params('J', {'f_max': 500})
        a.set_new_params('S', {'f_max': 700, 'alpha': 0.7})
        assert a.fjung_max == 500
        assert a.fsav_max == 700
        assert a.alpha == 0.7
        with pytest.raises(ValueError):
            a.set_new_params('J', {'f_max': -200})
        with pytest.raises(ValueError):
            a.set_new_params('S', {'f_max': -200})
        with pytest.raises(ValueError):
            a.set_new_params('S', {'alpha': -200})
        with pytest.raises(KeyError):
            a.set_new_params('Q', {'f_max': 500})
        with pytest.raises(KeyError):
            a.set_new_params('S', {'rasmus': 69})

    def test_set_food(self):
        a = self.simple_island()
        a.set_food((1, 3))
        b = Island("OOOOO\nOSSSO\nOSSSO\nOSSSO\nOOOOO")
        b.set_food((1, 3))
        assert a.food == {(1, 3): 800}
        assert b.food == {(1, 3): 300}

    def test_set_more_food(self):
        a = self.savannah_island()
        for i in range(3):
            for j in range(3):
                a.set_food((i, j))
        assert a.food[(1, 2)] == 300
        assert a.food[(1, 1)] == 300
        assert a.food[(0, 0)] == 0

    def test_grow_food_not_growing_while_full(self):
        a = self.simple_island()
        a.set_food((1, 1))
        a.grow_food((1, 1))
        assert a.food[1, 1] == 800

    def test_grow_food(self):
        a = self.savannah_island()
        b = Herbivores()
        a.set_food((1, 1))
        animals_to_add = [{'loc': (1, 1), 'pop': [{'species': 'Herbivore', 'age': 16, 'weight': 1356.3}]}]
        a.add_animals(animals_to_add)
        b.animals_eat((1, 1), a, a.herbs)
        a.grow_food((1, 1))
        assert a.food[1, 1] == 290 + 0.3 * (300 - 290)

    def test_multiple_grow_food(self):
        a = self.savannah_island()
        b = Herbivores()
        animals_to_add = [{'loc': (1, 2), 'pop': [{'species': 'Herbivore', 'age': 16, 'weight': 1356.3}]},
                          {'loc': (1, 1), 'pop': [{'species': 'Herbivore', 'age': 16, 'weight': 1356.3}]},
                          {'loc': (2, 2), 'pop': [{'species': 'Herbivore', 'age': 16, 'weight': 1356.3}]},
                          ]
        a.add_animals(animals_to_add)
        for i in range(3):
            for j in range(3):
                a.set_food((i, j))
        for i in range(3):
            for j in range(3):
                b.animals_eat((i, j), a, a.herbs)
                a.grow_food((i, j))
        assert a.food[1, 2] == (290 + 0.3 * (300 - 290))
        assert a.food[1, 1] == (290 + 0.3 * (300 - 290))
        assert a.food[2, 2] == (290 + 0.3 * (300 - 290))

    def test_add_animals_simple(self):
        added_list = [{'loc': (1, 1), 'pop': [{'species': 'Herbivore', 'age': 0.1, 'weight': 1.3}]}]
        a = self.simple_island()
        a.add_animals(added_list)
        assert a.herbs == {(1, 1): [{'species': 'Herbivore', 'age': 0.1, 'weight': 1.3}]}

    def test_add_animal_wrong_terain(self):
        added_list = [{'loc': (0, 0), 'pop': [{'species': 'Herbivore', 'age': 0.1, 'weight': 1.3}]}]
        a = self.simple_island()
        with pytest.raises(ValueError):
            a.add_animals(added_list)

    def test_add_animals_both(self):
        added_list = [{'loc': (1, 1), 'pop': [{'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3}]},
                      {'loc': (1, 1), 'pop': [{'species': 'Carnivore', 'age': 0.1, 'Weight': 1.3}]}]
        a = self.simple_island()
        a.add_animals(added_list)
        assert a.herbs == {(1, 1): [{'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3}]}
        assert a.carns == {(1, 1): [{'species': 'Carnivore', 'age': 0.1, 'Weight': 1.3}]}

    def test_add_complex_list(self):
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 19.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 107.3}]},
                      {'loc': (2, 2), 'pop': [{'species': 'Herbivore', 'age': 3, 'weight': 10},
                      {'species': 'Herbivore', 'age': 4, 'weight': 9},
                      {'species': 'Herbivore', 'age': 5, 'weight': 10}]},
                      {'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 1000, 'weight': 1000000.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 20.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 0.3}]}]
        a = self.savannah_island()
        a.add_animals(added_list)
        assert len(a.herbs[(3, 3)]) == 4
        assert len(a.carns[(3, 3)]) == 2

    def test_food_gets_eaten(self):
        a = self.savannah_island()
        a.set_food((3, 3))
        assert a.food_gets_eaten((3, 3), 200) == 200
        assert a.food_gets_eaten((3, 3), 200) == 100
        assert a.food_gets_eaten((3, 3), 200) == 0

    def test_number_of_animals(self):
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 19.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 107.3}]},
                      {'loc': (2, 2), 'pop': [{'species': 'Herbivore', 'age': 3, 'weight': 10},
                      {'species': 'Herbivore', 'age': 4, 'weight': 9},
                      {'species': 'Herbivore', 'age': 5, 'weight': 10}]},
                      {'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 1000, 'weight': 1000000.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 20.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 0.3}]}]
        a = self.savannah_island()
        a.add_animals(added_list)
        herbs, carns = a.num_animals()
        assert herbs == 7
        assert carns == 2


class TestAnimal:
    """
    Class to test the animal class.
    """
    def jungle_island_animals(self):
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                              {'species': 'Herbivore', 'age': 30, 'weight': 19.3},
                                              {'species': 'Herbivore', 'age': 10, 'weight': 107.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 20.3}]}]
        a = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
        a.add_animals(added_list)
        b = Herbivores()
        return a, b


    def test_aging(self):
        a, b = self.jungle_island_animals()
        b.aging((3, 3), a.herbs)
        assert a.herbs[(3, 3)][0]['age'] == 21
        assert a.herbs[(3, 3)][1]['age'] == 31
        assert a.herbs[(3, 3)][2]['age'] == 11
        assert a.carns[(3, 3)][0]['age'] == 1
        b.aging((3, 3), a.carns)
        assert a.carns[(3, 3)][0]['age'] == 2

    def test_death(self):
        a, b = self.jungle_island_animals()
        for _ in range(40):
            b.aging((3, 3), a.herbs)
            b.calculate_fitness((3, 3), a.herbs)
            b.death((3, 3), a.herbs)
        assert len(a.herbs[(3, 3)]) == 0
        assert len(a.carns[(3, 3)]) == 1

    def test_zero_weight_zero_fitness_equals_death(self):
        a, b = self.jungle_island_animals()
        a.add_animals([{'loc': (1, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 0}]}])
        b.calculate_fitness((1, 3), a.herbs)
        assert a.herbs[(1, 3)][0]['fitness'] == 0

    def test_not_dying_young_and_healthy(self):
        a, b = self.jungle_island_animals()
        b.calculate_fitness((3, 3), a.carns)
        b.death((3, 3), a.carns)
        assert len(a.carns[(3, 3)]) == 1


class TestHerbivores:
    def test_herbivores_call(self):
        """Default constructor callable"""
        a = Herbivores()
        assert isinstance(a, Herbivores)

    def test_default_values(self):
        a = Herbivores()
        assert a.w_birth == 8.0
        assert a.sigma_birth == 1.5
        assert a.beta == 0.9
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
        added_list = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(added_list, b)
        assert a.animal == {(3, 1): [{'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3}]}

    def test_add_animals_twice(self):
        added_list = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(added_list, b)
        a.add_animal(added_list, b)
        assert a.herbs == {(3, 1): [{'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3},
                                    {'species': 'Herbivore', 'age': 0.1, 'Weight': 1.3}]}

    def test_add_multiple_animals(self):
        added_list1 = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 16, 'Weight': 1356.3},
                                              {'species': 'Herbivore', 'age': 113, 'Weight': 1323}]}]
        added_list2 = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 12, 'Weight': 21.3},
                                              {'species': 'Herbivore', 'age': 123, 'Weight': 321.3}]},
                      {'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 166, 'Weight': 135.3},
                                              {'species': 'Herbivore', 'age': 11, 'Weight': 323}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(added_list1, b)
        a.add_animal(added_list2, b)
        assert a.herbs[(3, 1)] == [{'species': 'Herbivore', 'age': 16, 'Weight': 1356.3},
                                   {'species': 'Herbivore', 'age': 113, 'Weight': 1323},
                                   {'species': 'Herbivore', 'age': 166, 'Weight': 135.3},
                                   {'species': 'Herbivore', 'age': 11, 'Weight': 323}]
        assert a.herbs[(3, 3)] == [{'species': 'Herbivore', 'age': 12, 'Weight': 21.3},
                                   {'species': 'Herbivore', 'age': 123, 'Weight': 321.3}]

    def test_calculate_fitness(self):
        added_list = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 1, 'weight': 1.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 1.3}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(added_list, b)
        a.calculate_fitness((3, 1))
        assert abs(a.herbs[(3, 1)][0]['fitness'] - 0.2951333) < 0.001

    def test_sort_by_fitness(self):
        added_list1 = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 1, 'weight': 10.3}]},
                       {'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 6, 'weight': 16}]}]
        added_list2 = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 6, 'weight': 16}]},
                       {'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 1, 'weight': 10.3}]}]
        a = Herbivores()
        b = Herbivores()
        c = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(added_list1, c)
        a.calculate_fitness((3, 1))
        a.sort_by_fitness((3, 1))
        b.add_animal(added_list2, c)
        b.calculate_fitness((3, 1))
        b.sort_by_fitness((3, 1))
        assert a.herbs[(3, 1)][0]['fitness'] >= a.herbs[(3, 1)][1]['fitness']
        assert b.herbs[(3, 1)][0]['fitness'] >= b.herbs[(3, 1)][1]['fitness']

    def test_animals_eat(self):
        added_list1 = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 1, 'weight': 10.3}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        b.set_food(pos=(3, 1))
        a.add_animal(added_list1, b)
        abc = a.herbs[(3, 1)][0]['weight']
        a.animals_eat((3, 1), b)
        assert abc < a.herbs[(3, 1)][0]['weight']

    def test_breeding(self):
        population = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 2, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 3, 'weight': 29.3},
                                              {'species': 'Herbivore', 'age': 12, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 3, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 5, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 6, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 7, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 8, 'weight': 27.3},
                                              {'species': 'Herbivore', 'age': 9, 'weight': 27.3}
                                              ]}]
        a = Herbivores(seed=3)
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        b.set_food((3, 1))
        a.add_animal(population, b)
        a.animals_eat((3, 1), b)
        len_list1 = len(a.herbs[(3, 1)])
        a.calculate_fitness((3, 1))
        a.breeding((3, 1), b)
        len_list2 = len(a.herbs[(3, 1)])
        assert len_list2 > len_list1

    def test_aging(self):
        population = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 1, 'weight': 10.3}]},
                      {'loc': (2, 2), 'pop': [{'species': 'Herbivore', 'age': 6, 'weight': 16}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(population, b)
        a.aging((3, 1))
        assert a.herbs[(3, 1)][0]['age'] == 2
        assert a.herbs[(2, 2)][0]['age'] == 6
        a.aging((2, 2))
        assert a.herbs[(2, 2)][0]['age'] == 7

    def test_loss_of_weight(self):
        population = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 1, 'weight': 10.3}]},
                      {'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 6, 'weight': 16}]}]
        a = Herbivores()
        b = Island("OOOOO\nOSSSO\nOJJJO\nOSSJO\nOSSSO\nOOOOO")
        a.add_animal(population, b)
        a0 = a.herbs[(3, 1)][0]['weight']
        a1 = a.herbs[(3, 1)][1]['weight']
        a.loss_of_weight((3, 1))
        assert a0 > a.herbs[(3, 1)][0]['weight']
        assert a1 > a.herbs[(3, 1)][1]['weight']

    def test_death(self):
        population = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 10000, 'weight': 10.3}]},
                      {'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 60000, 'weight': 0}]}]
        a = Herbivores()
        b = Island("OOOOO\nOOOOO\nOJOOO\nOJOOO\nOOOOO\nOOOOO")
        a.add_animal(population, b)
        a.calculate_fitness((3, 1))
        initial_len = len(a.herbs[(3, 1)])
        a.death((3, 1))
        assert len(a.herbs[(3, 1)]) < initial_len

    def test_migration_calculation(self):
        population = [{'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 5, 'weight': 13.3}]},
                      {'loc': (3, 1), 'pop': [{'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10}]}]
        a = Island("OOOOO\nOOOOO\nOJOOO\nOJOOO\nOOOOO\nOOOOO")
        b = Herbivores(seed=1)
        b.add_animal(population, a)
        a.set_food((3, 1))
        a.set_food((2, 1))
        a.set_food((3, 2))
        a.set_food((3, 0))
        a.set_food((4, 1))
        b.calculate_fitness((3, 1))
        b.migration_calculations(a.rader, a.col, a)
        assert len(b.herbs[(3, 1)]) == 9
        assert len(b.idx_for_animals_to_remove) > 0
        b.migration_execution(a)
        assert len(b.herbs[(3, 1)]) == 7
        assert len(b.herbs[(2, 1)]) == 2

    def test_invalid_moving(self):
        population = [{'loc': (1, 1), 'pop': [{'species': 'Herbivore', 'age': 5, 'weight': 13.3}]},
                      {'loc': (1, 1), 'pop': [{'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10}]}]
        a = Herbivores(seed=1)
        b = Island("OOO\nOJO\nOOO")
        for i in range(b.rader):
            for j in range(b.col):
                b.set_food((i, j))
        a.add_animal(population, b)
        a.calculate_fitness((1, 1))
        a.migration_calculations(b.rader, b.col, b)
        a.migration_execution(b)
        assert (2, 1) not in a.herbs.keys()
        assert len(a.herbs[(1, 1)]) == 9

    def test_migration_carnivores(self):
        population = [{'loc': (3, 2), 'pop': [{'species': 'Herbivore', 'age': 5, 'weight': 13.3}]},
                      {'loc': (3, 2), 'pop': [{'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10}]}]
        population1 = [{'loc': (3, 1), 'pop': [{'species': 'Carnivore', 'age': 5, 'weight': 13.3}]},
                      {'loc': (3, 1), 'pop': [{'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                              {'species': 'Carnivore', 'age': 4, 'weight': 10}]}]
        a = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
        b = Herbivores(seed=1)
        c = Carnivores(seed=1)
        b.add_animal(population, a)
        a.add_carnivores(population1)
        a.set_food((3, 1))
        a.set_food((2, 1))
        a.set_food((3, 2))
        a.set_food((3, 0))
        a.set_food((4, 1))
        c.calculate_fitness((3, 1))
        c.migration_calculations(a.rader, a.col, a, b)
        assert len(c.carns[(3, 1)]) == 9
        assert (3, 2) not in d.carns.keys()
        assert len(c.idx_for_animals_to_remove) > 0
        print(c.idx_for_animals_to_remove)
        og_animal = c.carns[(3, 1)][2]
        c.migration_execution()
        assert og_animal in c.carns[(3, 2)]
        assert len(c.carns[(3, 2)]) == 3
        assert len(c.carns[(3, 1)]) == 6
