# -*- coding: utf-8 -*-

"""
This code will contain the island class
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

import numpy as np


class Island:
    def __init__(self, le_map=None):
        """Sets the variables in the Island-class"""
        self.valid_map_vals = ['O', 'D', 'M', 'S', 'J']
        self.le_map = le_map
        if self.le_map is None:
            self.le_map = "OOO\nOJO\nOOO"
        self.rader = None
        self.col = None
        Island.string_to_matrix(self)

    def string_to_matrix(self):
        """Converts the input multiline-string to a matrix"""
        if type(self.le_map) is not str:
            raise TypeError('Input needs to be a string')
        list1 = self.le_map.split()
        list2 = []
        for i in range(len(list1)):
            list2 += [hc for hc in list1[i]]

        self.col = int(len(list2) / len(list1))
        self.rader = int(len(list1))

        self.le_map = np.reshape(list2, (self.rader, self.col))

    def limit_map_vals(self):
        """Raises ValueErrors if the input island-string violates any of the criterions for the island"""
        i_max = self.rader - 1
        j_max = self.col - 1

        for i in range(self.rader):
            for j in range(self.col):
                if self.le_map[i, j] not in self.valid_map_vals:
                    raise ValueError('One or more of the terraintypes are not valid')
                if i == 0 or j == 0 or i == i_max or j == j_max:
                    if self.le_map[i, j] != 'O':
                        raise ValueError('One or more of the perimeter-tiles are not ocean')

    def fetch_map(self):
        """Returns the map"""
        return self.le_map

    def fetch_naturetype(self, pos):
        """Fetches the naturetype of the map in the input position"""
        return self.le_map[pos]


class Savannah:
    def __init__(self, fsav_max=None, alpha=None, f=None):
        """
        Class for the savannah tiles

        Input:
        :param fsav_max is the maximum amount of food on the savannah tiles
        :param alpha: The growing factor for the food
        :param f: the amount of food the herbivores eat if there is enough food.
        """
        self.food = {}
        self.fsav_max = fsav_max
        self.f = f
        self.alpha = alpha
        if fsav_max is None:
            self.fsav_max = 300

        if f is None:
            self.f = 10

        if alpha is None:
            self.alpha = 0.3

    def set_food(self, pos):
        """
        sets the intial amount of food for the savannah tiles

        :param pos: The position of the map
        :return:
        """
        self.food.update({pos: self.fsav_max})

    def grow_food(self, pos):
        """
        updates the amount of food after the animals have eaten
        :param pos: The position of the map
        """
        self.food.update({pos: self.food[pos] + self.alpha * (self.fsav_max - self.food[pos])})

    def food_gets_eaten(self, pos):
        """
        reduces the amount of food avaliable on the tiles

        :param pos: THe position of the map
        :return: gives out the amount of food eaten
        """
        if self.f <= self.food[pos]:
            self.food[pos] -= self.f
            return self.f
        elif self.food[pos] == 0:
            return 0
        else:
            b = self.food[pos]
            self.food[pos] = 0
            return b


class Jungle:
    def __init__(self, f_jungle=None):
        """
        The jungle class
        :param f_jungle:
        """
        self.food = {}
        self.f_jungle = f_jungle
        if self.f_jungle is None:
            self.f_jungle = 800

    def update_food(self, pos):
        """
        updates the food on the jungle tiles
        :param pos:
        :return:
        """
        self.food[pos] = self.f_jungle


class Desert:
    def __init__(self):
        """
        Class for dessert
        """
        self.food_animals_data = {}

    def set_food(self, pos):
        pass


class Animals:
    def __init__(self):
        pass


class Herbivores:
    def __init__(self, w_birth=8.0, sigma_birth=1.5, beta=0.9, eta=0.05, a_half=40.0, phi_age=0.2, w_half=10.0,
                 phi_weight=0.1, mu=0.25, lambda1=1.0, gamma=0.2, zeta=3.5, xi=1.2, omega=0.4, seed=1):
        """
        The class containing all the necessary functions for herbivores

        :param w_birth: The average weight for a newborn Herbivore
        :param sigma_birth: The standard deviation for a newborn
        :param beta: The growing factor telling how much of the food is changed into weight
        :param eta: The weight reduction factor
        :param a_half: Fitness-factor
        :param phi_age: Fitness-factor
        :param w_half: Fitness-factor
        :param phi_weight: Fitness-factor
        :param mu: ???????
        :param lambda1: Migration-factor
        :param gamma: gives the probability for giving birth, given number of animals on same tiles and their fitness
        :param zeta: Gives the restrictions for giving girth depending on weight
        :param xi: The factor for weight loss after given birth
        :param omega: the probability of dieing given the animals fitnessvalue
        """
        self.w_birth = w_birth
        self.sigma_birth = sigma_birth
        self.beta = beta
        self.eta = eta
        self.a_half = a_half
        self.phi_age = phi_age
        self.w_half = w_half
        self.phi_weight = phi_weight
        self.mu = mu
        self.lambda1 = lambda1
        self.gamma = gamma
        self.zeta = zeta
        self.xi = xi
        self.omega = omega
        self.herbs = {}
        self.seed = seed
        np.random.seed(self.seed)

    def add_animal(self, animal_list):
        """
        Adds animals to the map
        :param animal_list: A list that contains the animals wegiht, age and species and where we want to add them
        :return:
        """
        for animal in animal_list:
            if animal['loc'] not in self.herbs.keys():
                self.herbs.update({animal['loc']: animal['pop']})
            else:
                self.herbs[animal['loc']] += animal['pop']

    def calculate_fitness(self, pos):
        """
        Calculates the fitness for all the animals on one tile
        :param pos: gives which tile we want to calculate the fitness
        :return:
        """
        for animal in self.herbs[pos]:
            if animal['weight'] == 0:
                new_fitness = {'fitness': 0}
                animal.update(new_fitness)
            else:
                new_fitness = {'fitness': (1 / (1 + np.exp(self.phi_age * (animal['age'] - self.a_half)))) *
                                          (1 / (1 + np.exp(-(self.phi_weight * (animal['weight'] - self.w_half)))))}
                animal.update(new_fitness)

    def sort_by_fitness(self, pos):
        """
        Sorts the animals on a tile after their fitness
        :param pos: the position(tile)
        :return:
        """
        self.herbs[pos] = sorted(self.herbs[pos], key=lambda i: i['fitness'], reverse=True)

    def animals_eat(self, pos, terrain_class):
        """
        animals eat, in order of their fitness
        :param pos: the position/tile
        :param terrain_class: retrives either savannah class or jungle class, to extract food_gets_eat
        :return:
        """
        for idx, animal in enumerate(self.herbs[pos]):
            food = terrain_class.food_gets_eaten(pos)
            self.herbs[pos][idx]['weight'] += self.beta * food

    def breeding(self, pos):
        """
        breeds animal on the given tile, depending on the set parameters
        :param pos: the position/tile
        :return:
        """
        children = []
        n = len(self.herbs[pos])
        for idx, animal in enumerate(self.herbs[pos]):
            if animal['weight'] < self.zeta * (self.w_birth + self.sigma_birth):
                p = 0
            else:
                p = min(1, self.gamma * animal['fitness'] * (n - 1))
            if p > np.random.rand(1):
                w = np.random.normal(self.w_birth, self.sigma_birth)
                if animal['weight'] > self.xi * w:
                    children.append({'loc': pos, 'pop': [{'species': 'Herbievore', 'age': 0, 'weight': w}]})
                    self.herbs[pos][idx]['weight'] -= self.xi * w
        if len(children) > 0:
            Herbivores.add_animal(self, children)

    def aging(self, pos):
        """
        ages all the animal on one tile with 1 year
        :param pos: the position/tile
        :return:
        """
        for idx in range(len(self.herbs[pos])):
            self.herbs[pos][idx]['age'] += 1

    def loss_of_weight(self, pos):
        """
        reduces the weight of all the animals on a single tile
        :param pos: the position/tile
        :return:
        """
        for idx in range(len(self.herbs[pos])):
            self.herbs[pos][idx]['weight'] -= self.eta * self.herbs[pos][idx]['weight']

    def death(self, pos):
        """
        removes the animals the died the last year from the list
        :param pos: the position asked for
        :return:
        """

        a = []
        for idx, animal in enumerate(self.herbs[pos]):
            if animal['fitness'] == 0:
                a.append(idx)
            else:
                p = self.omega * (1 - animal['fitness'])
                if p >= np.random.rand(1):
                    a.append(idx)
        for idx in sorted(a, reverse=True):
            del self.herbs[pos][idx]


class Carnivores:
    def __init__(self):
        pass
