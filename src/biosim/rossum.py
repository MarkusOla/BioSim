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

        self.le_map = np.reshape(list2, (self.col, self.rader))

    def limit_map_vals(self):
        """Raises ValueErrors if the input island-string violates any of the criterions for the island"""
        i_max = self.rader - 1
        j_max = self.col - 1

        for i in range(self.rader):
            for j in range(self.col):
                if self.le_map[i, j] not in self.valid_map_vals:
                    raise ValueError('One or more of the terraintypes are not valid')
                elif i == 0 or j == 0 or i == i_max or j == j_max:
                    if self.le_map[i, j] != 'O':
                        raise ValueError('One or more of the perimeter-tiles are not ocean')

    def fetch_map(self):
        """Returns the map"""
        return self.le_map

    def fetch_naturetype(self, pos):
        """Fetches the naturetype of the map in the input position"""
        return self.le_map[pos]


class Savannah:
    """Class for the savannah tiles"""
    def __init__(self, fsav_max=None, alpha=None, f=None):
        self.food = {}
        if fsav_max is None:
            self.fsav_max = 300
        else:
            self.fsav_max = fsav_max
        if f is None:
            self.f = 10
        else:
            self.f = f

        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = alpha

    def set_food(self, pos):
        self.food.update({pos: self.fsav_max})

    def grow_food(self, pos):
        self.food.update({pos: self.food[pos] + self.alpha * (self.fsav_max - self.food[pos])})

    def food_gets_eaten(self, pos):
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
        self.food = {}
        self.f_jungle = f_jungle
        if self.f_jungle is None:
            self.f_jungle = 800

    def update_food(self, pos):
        self.food[pos] = self.f_jungle


class Desert:
    def __init__(self):
        self.food_animals_data = {}

    def set_food(self, pos):
        pass


class Animals:
    def __init__(self):
        pass


class Herbivores:
    def __init__(self, w_birth=8.0, sigma_birth=8.0, beta=1.5, eta=0.05, a_half=40.0, phi_age=0.2, w_half=10.0,
                 phi_weight=0.1, mu=0.25, lambda1=1.0, gamma=0.2, zeta=3.5, xi=1.2, omega=0.4):
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

    def add_animal(self, animal_list):
        for animal in animal_list:
            if animal['loc'] not in self.herbs.keys():
                self.herbs.update({animal['loc']: animal['pop']})
            else:
                self.herbs[animal['loc']] += animal['pop']

    def calculate_fitness(self, pos):
        for animal in self.herbs[pos]:
            if animal['weight'] == 0:
                new_fitness = {'fitness': 0}
                animal.update(new_fitness)
            else:
                new_fitness = {'fitness': (1 / (1 + np.exp(self.phi_age * (animal['age'] - self.a_half)))) *
                                          (1 / (1 + np.exp(self.phi_weight * (animal['weight'] - self.w_half))))}
                animal.update(new_fitness)

    def sort_by_fitness(self, pos):
        self.herbs[pos] = sorted(self.herbs[pos], key = lambda i: i['fitness'], reverse=True)

    def animals_eat(self, pos):
        for idx, animal in enumerate(self.herbs[pos]):
            food = Savannah.food_gets_eaten(pos)
            self.herbs[pos][idx]['weigth'] += self.beta * food

    def breeding(self, pos):
        children = []
        N = len(self.herbs[pos])
        for idx, animal in enumerate(self.herbs[pos]):
            if animal['weight'] < self.zeta * (self.w_birth + self.sigma_birth):
                p = 0
            else:
                p = min(1, self.gamma * animal['fitness'] * (N - 1))
            if p > np.random.rand(1):
                w = np.random.normal(self.w_birth, self.sigma_birth)
                if animal['weight'] > self.xi * w
                    children.append({'species': 'Herbivore', 'age': 0, 'weight': w})
                    self.herbs[pos][idx]['weigth'] -= self.xi * w
        self.herbs[pos].append(children)






    def aging(self, pos):
        pass

    def loss_of_weight(self, pos):
        pass

    def death(self):
        pass


class Carnivores:
    def __init__(self):
        pass
