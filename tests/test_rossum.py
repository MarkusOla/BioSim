# -*- coding: utf-8 -*-

"""
This file will test the island.py-file
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

from biosim.island import Island
from biosim.animal import Herbivores, Carnivores
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
    def test_default_call(self):
        """
        Testing that that each of the two animalclasses are callable and that the default values are set properly
        """
        a = Herbivores()
        b = Carnivores()
        assert a.w_birth == 8
        assert a.sigma_birth == 1.5
        assert a.beta == 0.9
        assert a.eta == 0.05
        assert a.a_half == 40
        assert a.phi_age == 0.2
        assert a.w_half == 10
        assert a.phi_weight == 0.1
        assert a.mu == 0.25
        assert a.lambda1 == 1
        assert a.gamma == 0.2
        assert a.zeta == 3.5
        assert a.xi == 1.2
        assert a.omega == 0.4
        assert a.f == 10
        assert b.w_birth == 6
        assert b.sigma_birth == 1.0
        assert b.beta == 0.75
        assert b.eta == 0.125
        assert b.a_half == 60
        assert b.phi_age == 0.4
        assert b.w_half == 4
        assert b.phi_weight == 0.4
        assert b.mu == 0.4
        assert b.lambda1 == 1
        assert b.gamma == 0.8
        assert b.zeta == 3.5
        assert b.xi == 1.1
        assert b.omega == 0.9
        assert b.f == 50
        assert b.deltaphimax == 10

    def test_set_new_params_raise_errors(self):
        a = Herbivores()
        b = Carnivores()
        with pytest.raises(TypeError):
            a.set_new_params([1243])
        with pytest.raises(TypeError):
            b.set_new_params([1234])
        with pytest.raises(KeyError):
            a.set_new_params({'W_BIRTH': 1})
        with pytest.raises(KeyError):
            b.set_new_params({'W_BIRTH': 1})
        with pytest.raises(ValueError):
            a.set_new_params({'w_birth': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'w_birth': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'sigma_birth': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'sigma_birth': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'beta': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'beta': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'eta': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'eta': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'a_half': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'a_half': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'phi_age': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'phi_age': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'w_half': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'w_half': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'phi_weight': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'phi_weight': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'mu': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'mu': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'lambda': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'lambda': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'gamma': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'gamma': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'zeta': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'zeta': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'xi': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'xi': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'omega': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'omega': -1})
        with pytest.raises(ValueError):
            a.set_new_params({'F': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'F': -1})
        with pytest.raises(ValueError):
            b.set_new_params({'DeltaPhiMax': -1})

    def test_set_new_params(self):
        a = Herbivores()
        b = Carnivores()
        a.set_new_params({'w_birth': 12.0,
                          'sigma_birth': 1.3,
                          'beta': 1,
                          'eta': 0.06,
                          'a_half': 41.0,
                          'phi_age': 0.3,
                          'w_half': 11.0,
                          'phi_weight': 0.2,
                          'mu': 0.3,
                          'lambda': 1.1,
                          'gamma': 0.3,
                          'zeta': 3.6,
                          'xi': 1.3,
                          'omega': 0.5,
                          'F': 11.0})
        b.set_new_params({'w_birth': 8.0,
                          'sigma_birth': 1.2,
                          'beta': 0.9,
                          'eta': 0.05,
                          'a_half': 40.0,
                          'phi_age': 0.2,
                          'w_half': 10.0,
                          'phi_weight': 0.1,
                          'mu': 0.25,
                          'lambda': 1.0,
                          'gamma': 0.2,
                          'zeta': 3.5,
                          'xi': 1.2,
                          'omega': 0.4,
                          'F': 10.0,
                          'DeltaPhiMax': 11})
        assert a.w_birth == 12
        assert a.sigma_birth == 1.3
        assert a.beta == 1
        assert a.eta == 0.06
        assert a.a_half == 41
        assert a.phi_age == 0.3
        assert a.w_half == 11
        assert a.phi_weight == 0.2
        assert a.mu == 0.3
        assert a.lambda1 == 1.1
        assert a.gamma == 0.3
        assert a.zeta == 3.6
        assert a.xi == 1.3
        assert a.omega == 0.5
        assert a.f == 11
        assert b.w_birth == 8
        assert b.sigma_birth == 1.2
        assert b.beta == 0.9
        assert b.eta == 0.05
        assert b.a_half == 40
        assert b.phi_age == 0.2
        assert b.w_half == 10
        assert b.phi_weight == 0.1
        assert b.mu == 0.25
        assert b.lambda1 == 1.0
        assert b.gamma == 0.2
        assert b.zeta == 3.5
        assert b.xi == 1.2
        assert b.omega == 0.4
        assert b.f == 10
        assert b.deltaphimax == 11

    def jungle_island_animals(self):
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 21, 'weight': 17.3},
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
        assert a.herbs[(3, 3)][0]['age'] == 22
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

    def test_breeding_herbivores(self):
        a, b = self.jungle_island_animals()
        len_list1 = len(a.herbs[(3, 3)])
        b.calculate_fitness((3, 3), a.herbs)
        for _ in range(10):
            b.breeding((3, 3), a, a.herbs)
        len_list2 = len(a.herbs[(3, 3)])
        assert len_list2 > len_list1

    def test_breeding_carnivores(self):
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Carnivore', 'age': 20, 'weight': 17.3},
                                              {'species': 'Carnivore', 'age': 30, 'weight': 19.3},
                                              {'species': 'Carnivore', 'age': 10, 'weight': 107.3},
                                              {'species': 'Carnivore', 'age': 1, 'weight': 20.3}]}]
        a = self.jungle_island_animals()[0]
        b = Carnivores()
        a.set_food((3, 3))
        a.add_animals(added_list)
        len_list1 = len(a.carns[(3, 3)])
        b.calculate_fitness((3, 3), a.carns)
        for _ in range(10):
            b.breeding((3, 3), a, a.carns)
        len_list2 = len(a.carns[(3, 3)])
        assert len_list2 > len_list1

    def test_loss_of_weight(self):
        a, b = self.jungle_island_animals()
        weight0 = a.herbs[(3, 3)][0]['weight']
        weight1 = a.herbs[(3, 3)][1]['weight']
        b.loss_of_weight((3,3), a.herbs)
        assert abs(a.herbs[(3, 3)][0]['weight'] - weight0 * (1 - b.eta)) < 0.0001
        assert abs(a.herbs[(3, 3)][1]['weight'] - weight1 * (1 - b.eta)) < 0.0001

    def test_sort_by_fitness(self):
        a, b = self.jungle_island_animals()
        b.calculate_fitness((3, 3), a.herbs)
        assert a.herbs[(3, 3)][0]['fitness'] < a.herbs[(3, 3)][2]['fitness']
        b.sort_by_fitness((3, 3), a.herbs)
        assert a.herbs[(3, 3)][0]['fitness'] > a.herbs[(3, 3)][2]['fitness']


class TestHerbivores:
    """
    Class to test the functions in the Herbivore-class
    """
    def jungle_island_animals(self):
        """
        Function to create an island that can be used for testing the Herbivore-class
        :return: Returns the Island-class with a map and animals and the Herbivore-class
        """
        added_list = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 21, 'weight': 17.3},
                                              {'species': 'Herbivore', 'age': 30, 'weight': 19.3},
                                              {'species': 'Herbivore', 'age': 10, 'weight': 107.3},
                                              {'species': 'Herbivore', 'age': 1, 'weight': 20.3}]}]
        a = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
        a.add_animals(added_list)
        b = Herbivores()
        return a, b

    def test_sort_before_getting_hunted(self):
        """
        Tests that the Herbivores are sorted from weakest to strongest before they get hunted by the carnivores
        """
        a, b = self.jungle_island_animals()
        b.calculate_fitness((3, 3), a.herbs)
        assert a.herbs[(3, 3)][0]['fitness'] > a.herbs[(3, 3)][1]['fitness']
        b.sort_before_getting_hunted((3, 3), a.herbs)
        assert a.herbs[(3, 3)][0]['fitness'] < a.herbs[(3, 3)][1]['fitness']

    def test_animals_eat(self):
        a, b = self.jungle_island_animals()
        a.set_food(pos=(3, 3))
        w0 = a.herbs[(3, 3)][0]['weight']
        b.animals_eat((3, 3), a, a.herbs)
        assert a.herbs[(3, 3)][0]['weight'] == w0 + b.beta * b.f

    def test_migration(self):
        """
        Tests that two of the animals gets moved with the given inputs and seed
        """
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
        b = Herbivores()
        a.add_animals(population)
        a.set_food((3, 1))
        a.set_food((2, 1))
        a.set_food((3, 2))
        a.set_food((3, 0))
        a.set_food((4, 1))
        b.calculate_fitness((3, 1), a.herbs)
        b.migration_calculations(a, a.herbs)
        assert len(a.herbs[(3, 1)]) == 9
        assert len(b.idx_for_animals_to_remove) > 0
        b.migration_execution(a, a.herbs)
        assert len(a.herbs[(3, 1)]) == 7
        assert len(a.herbs[(2, 1)]) == 2

    def test_invalid_moving(self):
        population = [{'loc': (2, 2), 'pop': [{'species': 'Herbivore', 'age': 5, 'weight': 13.3}]},
                      {'loc': (2, 2), 'pop': [{'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10},
                                              {'species': 'Herbivore', 'age': 4, 'weight': 10}]}]
        a = Island("OOOOO\nOOOOO\nOOJOO\nOOMOO\nOOOOO")
        b = Herbivores()
        for i in range(a.rader):
            for j in range(a.col):
                a.set_food((i, j))
        a.add_animals(population)
        b.calculate_fitness((2, 2), a.herbs)
        for _ in range(30):
            b.migration_calculations(a, a.herbs)
            b.migration_execution(a, a.herbs)
        assert (2, 1) not in a.herbs.keys()
        assert len(a.herbs[(2, 2)]) == 9

    def test_more_migration(self):
        pass

    def test_tot_weight_herbivores(self):
        """Test to test the tot_weight_herbivores-function both for a position with animals and an empty position"""
        a, b = self.jungle_island_animals()
        assert b.tot_weight_herbivores((3, 3), a.herbs) == 17.3 + 19.3 + 107.3 + 20.3
        assert b.tot_weight_herbivores((1, 1), a.herbs) == 0



class TestCarnivores:
    """
    Class to test the functions in the Carnivore-class
    """
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
        b = Carnivores()
        c = Herbivores()
        a.add_animals(population)
        a.add_animals(population1)
        b.calculate_fitness((3, 1), a.carns)
        b.migration_calculations(a, c, a.carns)
        assert len(a.carns[(3, 1)]) == 9
        assert (3, 2) not in a.carns.keys()
        assert len(b.idx_for_animals_to_remove) > 0
        b.migration_execution(a, a.carns)
        assert len(a.carns[(3, 2)]) > 0
        assert len(a.carns[(3, 1)]) < 9

    def test_invalid_moving_carns(self):
        """ Tests invalid moving is not happening"""
        population1 = [{'loc': (2, 2), 'pop': [{'species': 'Carnivore', 'age': 5, 'weight': 13.3}]},
                       {'loc': (2, 2), 'pop': [{'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10},
                                               {'species': 'Carnivore', 'age': 4, 'weight': 10}]}]
        a = Island("OOOOO\nOOMOO\nOOJOO\nOOMOO\nOOJOO")
        b = Carnivores()
        c = Herbivores()
        a.add_animals(population1)
        b.calculate_fitness((2, 2), a.carns)
        b.migration_calculations(a, c, a.carns)
        assert len(a.carns[(2, 2)]) == 9
        assert (3, 2) not in a.carns.keys()
        assert len(b.idx_for_animals_to_remove) == 0
        b.migration_execution(a, a.carns)
        assert len(a.carns[(2, 2)]) == 9

    def test_carnivore_leaves_food(self):
        """Testing that the carnivores stops eating when it has received f amount of food"""
        animal_list = [{'loc': (1, 1), 'pop': [{'species': 'Carnivore', 'age': 5, 'weight': 53.3},
                                               {'species': 'Carnivore', 'age': 5, 'weight': 53.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 13.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 13.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 13.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 13.3}]}]
        a = Island('OOO\nOJO\nOOO')
        b = Carnivores()
        c = Herbivores()
        a.add_animals(animal_list)
        b.set_new_params({'beta': 1, 'F': 5, 'DeltaPhiMax': 0.0001})
        c.calculate_fitness((1, 1), a.herbs)
        b.calculate_fitness((1, 1), a.carns)
        b.carnivores_eat((1, 1), a, a.carns)
        assert len(a.herbs[(1, 1)]) == 2
        assert a.carns[(1, 1)][0]['weight'] == 58.3
        assert a.carns[(1, 1)][1]['weight'] == 58.3

    def test_carnivore_leaves_food_and_eats_F(self):
        """Testing that the carnivores stops eating when it has received f amount of food"""
        animal_list = [{'loc': (1, 1), 'pop': [{'species': 'Carnivore', 'age': 5, 'weight': 53.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 23.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 23.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 23.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 23.3}]}]
        a = Island('OOO\nOJO\nOOO')
        b = Carnivores()
        c = Herbivores()
        a.add_animals(animal_list)
        b.set_new_params({'beta': 1, 'F': 50, 'DeltaPhiMax': 0.0001})
        c.calculate_fitness((1, 1), a.herbs)
        b.calculate_fitness((1, 1), a.carns)
        b.carnivores_eat((1, 1), a, a.carns)
        assert len(a.herbs[(1, 1)]) == 1
        assert a.carns[(1, 1)][0]['weight'] == 103.3

    def test_carnivore_does_not_eat_with_lower_fitness(self):
        """Testing that the carnivores stops eating when it has received f amount of food"""
        animal_list = [{'loc': (1, 1), 'pop': [{'species': 'Carnivore', 'age': 1, 'weight': 53.3},
                                               {'species': 'Herbivore', 'age': 1, 'weight': 23.3},
                                               {'species': 'Herbivore', 'age': 1, 'weight': 23.3},
                                               {'species': 'Herbivore', 'age': 1, 'weight': 23.3},
                                               {'species': 'Herbivore', 'age': 1, 'weight': 23.3}]}]
        a = Island('OOO\nOJO\nOOO')
        b = Carnivores()
        c = Herbivores()
        a.add_animals(animal_list)
        b.set_new_params({'beta': 1, 'F': 50})
        c.calculate_fitness((1, 1), a.herbs)
        b.calculate_fitness((1, 1), a.carns)
        for _ in range(40):
            b.carnivores_eat((1, 1), a, a.carns)
        assert len(a.herbs[(1, 1)]) < 4

    def test_not_enough_herbs_to_eat(self):
        animal_list = [{'loc': (1, 1), 'pop': [{'species': 'Carnivore', 'age': 5, 'weight': 53.3},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 20},
                                               {'species': 'Herbivore', 'age': 500, 'weight': 20}]}]
        a = Island('OOO\nOJO\nOOO')
        b = Carnivores()
        c = Herbivores()
        a.add_animals(animal_list)
        b.set_new_params({'beta': 1, 'F': 50, 'DeltaPhiMax': 0.0001})
        c.calculate_fitness((1, 1), a.herbs)
        b.calculate_fitness((1, 1), a.carns)
        b.carnivores_eat((1, 1), a, a.carns)
        assert len(a.herbs[(1, 1)]) == 0
        assert a.carns[(1, 1)][0]['weight'] == 93.3


