# -*- coding: utf-8 -*-

"""
This code will contain the island class
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

import numpy as np


class Island:
    def __init__(self, le_map=None, fsav_max=None, fjung_max=None, alpha=None):
        """
        Sets the variables in the Island-class

        Input:
        :param le_map is the string input of the map
        :param fsav_max is the maximum amount of food on the savannah tiles
        :param alpha: The growing factor for the food
        :param f: the amount of food the herbivores eat if there is enough food.
        """
        self.valid_map_vals = ['O', 'D', 'M', 'S', 'J']
        self.le_map = le_map
        self.rader = None
        self.col = None
        Island.string_to_matrix(self)
        self.herbs = {}
        self.carns = {}
        self.food = {}
        self.fsav_max = fsav_max
        self.alpha = alpha
        self.fjung_max = fjung_max

        if self.le_map is None:
            self.le_map = "OOO\nOJO\nOOO"
        if fsav_max is None:
            self.fsav_max = 300

        if alpha is None:
            self.alpha = 0.3

        if fjung_max is None:
            self.fjung_max = 800

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

    def set_new_params(self, new_params):
        """
        Set class parameters.
        Parameters
        ----------
        new_params : dict
            Legal keys: 'fsav_max', 'alpha', 'fjung_max'
        Raises
        ------
        ValueError, KeyError
        """
        default_params = {'fsav_max': 300.0,
                          'alpha': 0.3,
                          'fjung_max': 800}

        for key in new_params:
            if key not in (default_params.keys()):
                raise KeyError('Invalid parameter name: ' + key)

        if 'fsav_max' in new_params:
            if not 0 <= new_params['fsav_max']:
                raise ValueError('fsav_max must be larger or equal to 0')
            self.fsav_max = new_params['fsav_max']

        if 'fjung_max' in new_params:
            if not 0 <= new_params['fjung_max']:
                raise ValueError('fjung_max must be larger or equal to 0')
            self.fjung_max = new_params['fjung_max']

        if 'alpha' in new_params:
            if not 0 <= new_params['alpha']:
                raise ValueError('alpha must be larger or equal to 0.')
            self.alpha = new_params['alpha']

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

    def fetch_naturetype(self, pos):
        """Fetches the naturetype of the map in the input position"""
        return self.le_map[pos]

    def set_food(self, pos):
        """
        Sets the initial food-values for a tile of a given position
        :param pos: The position of the map
        """
        if self.fetch_naturetype(pos) == 'S':
            self.food.update({pos: self.fsav_max})
        elif self.fetch_naturetype(pos) == 'J':
            self.food.update({pos: self.fjung_max})
        else:
            self.food.update({pos: 0})

    def grow_food(self, pos):
        """
        Updates the amount of food after the animals have eaten in a given position
        :param pos: The position of the map
        """
        if self.fetch_naturetype(pos) == 'S':
            self.food.update({pos: self.food[pos] + self.alpha * (self.fsav_max - self.food[pos])})
        elif self.fetch_naturetype(pos) == 'J':
            self.food.update({pos: self.fjung_max})

    def food_gets_eaten(self, pos, f_animal):
        """
        reduces the amount of food avaliable on the tiles

        :param pos: THe position of the map
        :return: gives out the amount of food eaten
        """
        if f_animal <= self.food[pos]:
            self.food[pos] -= f_animal
            return f_animal
        elif self.food[pos] == 0:
            return 0
        else:
            b = self.food[pos]
            self.food[pos] = 0
            return b

    def add_animals(self, animal_list):
        """
       Adds herbivore to the map
        :param animal_list: A list that contains the animals wegiht, age and species and where we want to add them
        :return:
       """
        for dict in animal_list:
            for animal in dict['pop']:
                if self.fetch_naturetype(dict['loc']) == 'O' or \
                        self.fetch_naturetype(dict['loc']) == 'M':
                    raise ValueError('You are trying to put animals on ocean- or mountain-tiles')

                if animal['species'] == 'Herbivore':
                    if dict['loc'] not in self.herbs.keys():
                        self.herbs.update({dict['loc']: [animal]})
                    else:
                        self.herbs[dict['loc']].append(animal)
                else:
                    if dict['loc'] not in self.carns.keys():
                        self.carns.update({dict['loc']: [animal]})
                    else:
                        self.carns[dict['loc']].append(animal)


if __name__ == "__main__":
    herbivores = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]},
                  {'loc': (1, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]}
                  ]
    animalsdiff = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Carnivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]},
                  {'loc': (2, 3), 'pop': [{'species': 'Carnivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Carnivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]}
                  ]

    a = Cell()
    b = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
    c = Herbivores()
    a.add_animals(herbivores, b)
    a.add_animals(animalsdiff, b)

    c.calculate_fitness((3, 3), a.herbs)
    print(a.herbs)
    print(a.herbs[(3, 3)])
    print(a.carns)

class Herbivores:
    def __init__(self, w_birth=8.0, sigma_birth=1.5, beta=0.9, eta=0.05, a_half=40.0, phi_age=0.2, w_half=10.0,
                 phi_weight=0.1, mu=0.25, lambda1=1.0, gamma=0.2, zeta=3.5, xi=1.2, omega=0.4, seed=1, f = 10.0):
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
        self.f = f
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []

    def add_animal(self, animal_list, island_class):
        """
       Adds herbivore to the map
        :param animal_list: A list that contains the animals wegiht, age and species and where we want to add them
        :return:
       """
        for animal in animal_list:
            if island_class.fetch_naturetype(animal['loc']) == 'O' or \
                    island_class.fetch_naturetype(animal['loc']) == 'M':
                raise ValueError('You are trying to put animals on ocean- or mountain-tiles')
            if animal['loc'] not in self.herbs.keys():
                self.herbs.update({animal['loc']: animal['pop']})
            else:
                self.herbs[animal['loc']] += animal['pop']

    def calculate_fitness(self, pos):
        """
        Calculates the fitness for all the herbivores on one tile
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
        Sorts the herbivores on a tile after their fitness
        :param pos: the position(tile)
        :return:
        """
        self.herbs[pos] = sorted(self.herbs[pos], key=lambda i: i['fitness'], reverse=True)

    def sort_before_getting_hunted(self, pos):
        """
        Sorts the herbivores from worst to best fitness
        :param pos:
        :return:
        """
        self.herbs[pos] = sorted(self.herbs[pos], key=lambda i: i['fitness'])

    def animals_eat(self, pos, food_class):
        """
        herbivores eat, in order of their fitness
        :param pos: the position/tile
        :param food_class: retrives the fodder class, to make use of the food_gets_eat function
        :return:
        """
        for idx, animal in enumerate(self.herbs[pos]):
            food = food_class.food_gets_eaten(pos, self.f)
            self.herbs[pos][idx]['weight'] += self.beta * food

    def breeding(self, pos, island_class):
        """
        breeds herbivores on the given tile, depending on the set parameters
        :param pos: the position/tile
        :param island_class: the island, is used as in
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
            Herbivores.add_animal(self, children, island_class)

    def migration_calculations(self, rader, kolonner, island_class, food_class):
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for rad in range(1, rader - 1):
            for kol in range(1, kolonner - 1):
                pos = (rad, kol)
                if pos in self.herbs.keys():
                    for idx, animal in enumerate(self.herbs[pos]):
                        if animal['fitness'] * self.mu >= np.random.rand(1):
                            if (rad + 1, kol) in self.herbs.keys():
                                e_down = food_class.food[(rad + 1, kol)] / ((len(self.herbs[(rad + 1, kol)]) + 1) * self.f)
                            else:
                                e_down = food_class.food[(rad + 1, kol)] / self.f
                            if island_class.fetch_naturetype((rad + 1, kol)) == 'O' or island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                                p_down = 0
                            else:
                                p_down = np.exp(self.gamma * e_down)

                            if (rad - 1, kol) in self.herbs.keys():
                                e_up = food_class.food[(rad - 1, kol)] / ((len(self.herbs[(rad - 1, kol)]) + 1) * self.f)
                            else:
                                e_up = food_class.food[(rad - 1, kol)] / self.f
                            if island_class.fetch_naturetype((rad - 1, kol)) == 'O' or island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                                p_up = 0
                            else:
                                p_up = np.exp(self.gamma * e_up)

                            if (rad, kol - 1) in self.herbs.keys():
                                e_left = food_class.food[(rad, kol - 1)] / (
                                            (len(self.herbs[(rad, kol - 1)]) + 1) * self.f)
                            else:
                                e_left = food_class.food[(rad, kol - 1)] / self.f
                            if island_class.fetch_naturetype((rad, kol - 1)) == 'O' or island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                                p_left = 0
                            else:
                                p_left = np.exp(self.gamma * e_left)

                            if (rad, kol + 1) in self.herbs.keys():
                                e_right = food_class.food[(rad, kol + 1)] / (
                                            (len(self.herbs[(rad, kol + 1)]) + 1) * self.f)
                            else:
                                e_right = food_class.food[(rad, kol + 1)] / self.f
                            if island_class.fetch_naturetype((rad, kol + 1)) == 'O' or island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                                p_right = 0
                            else:
                                p_right = np.exp(self.gamma * e_right)

                            if p_up + p_right + p_left + p_down == 0:
                                break

                            prob_up = p_up / (p_down + p_left + p_right + p_up)
                            prob_down = p_down / (p_down + p_left + p_right + p_up)
                            prob_left = p_left / (p_down + p_left + p_right + p_up)
                            prob_right = p_right / (p_down + p_left + p_right + p_up)

                            direction = np.random.choice(np.arange(1, 5), p=[prob_right, prob_up, prob_left, prob_down])

                            if direction == 1:
                                self.animals_with_new_pos.append({'loc': (rad, kol + 1), 'pop': [animal]})
                            if direction == 2:
                                self.animals_with_new_pos.append({'loc': (rad - 1, kol), 'pop': [animal]})
                            if direction == 3:
                                self.animals_with_new_pos.append({'loc': (rad, kol - 1), 'pop': [animal]})
                            if direction == 4:
                                self.animals_with_new_pos.append({'loc': (rad + 1, kol), 'pop': [animal]})

                            self.idx_for_animals_to_remove.append([pos, idx])

    def migration_execution(self, island_class):
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del self.herbs[info[0]][info[1]]
        self.add_animal(self.animals_with_new_pos, island_class)

    def aging(self, pos):
        """
        ages all the herbivores on one tile with 1 year
        :param pos: the position/tile
        :return:
        """
        for idx in range(len(self.herbs[pos])):
            self.herbs[pos][idx]['age'] += 1

    def loss_of_weight(self, pos):
        """
        Reduces the weight of all the herbivores on a single tile
        :param pos: the position/tile
        :return:
        """
        for idx in range(len(self.herbs[pos])):
            self.herbs[pos][idx]['weight'] -= self.eta * self.herbs[pos][idx]['weight']

    def death(self, pos):
        """
        removes herbivores from the list according to the formula for death
        :param pos: the position asked for
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

    def tot_weight_herbivores(self, pos):
        if pos in self.herbs.keys():
            tot_weight = 0
            for herb in self.herbs[pos]:
                tot_weight += herb['weight']
        else:
            tot_weight = 0
        return tot_weight


class Carnivores:
    def __init__(self, w_birth=16.0, sigma_birth=1.0, beta=0.75, eta=0.125, a_half=60.0, phi_age=0.4, w_half=4.0,
                 phi_weight=0.4, mu=0.4, lambda1=1.0, gamma=0.8, zeta=3.5, xi=1.1, omega=0.9, f=50.0, DeltaPhiMax=10.0,
                 seed=1):
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
        self.f = f
        self.DeltaPhiMax = DeltaPhiMax
        self.carns = {}
        self.seed = seed
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        np.random.seed(self.seed)

    def add_carnivores(self, animal_list):
        """
        Adds carnivores to the map according to the input list
        :param animal_list: A list of which animals to put in and where they should be put in
        :return:
        """
        for animal in animal_list:
            if animal['loc'] not in self.carns.keys():
                self.carns.update({animal['loc']: animal['pop']})
            else:
                self.carns[animal['loc']] += animal['pop']

    def calculate_fitness(self, pos):
        """
        Calculates the fitness for all the carnivores on one tile
        :param pos: gives which tile we want to calculate the fitness
        :return:
        """
        if pos in self.carns.keys():
            for animal in self.carns[pos]:
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
        if pos in self.carns.keys():
            self.carns[pos] = sorted(self.carns[pos], key=lambda i: i['fitness'], reverse=True)

    def carnivores_eat(self, pos, herbivore_class):
        if pos in self.carns.keys():
            for idx1, carnivore in enumerate(self.carns[pos]):
                prey_weight = 0
                a = []
                for idx2, herbivore in enumerate(herbivore_class.herbs[pos]):
                    if carnivore['fitness'] <= herbivore['fitness']:
                        p = 0
                    elif carnivore['fitness'] - herbivore['fitness'] < self.DeltaPhiMax:
                        p = (carnivore['fitness'] - herbivore['fitness']) / self.DeltaPhiMax
                    else:
                        p = 1
                    if p > np.random.rand(1):
                        prey_weight += herbivore['weight']
                        a.append(idx2)
                    if prey_weight > self.f:
                        self.carns[pos][idx1]['weight'] += self.beta * self.f
                        for idx in sorted(a, reverse=True):
                            del herbivore_class.herbs[pos][idx]
                        break
                    elif prey_weight > 0 & idx2 == len(herbivore_class.herbs[pos]):
                        self.carns[pos][idx1]['weight'] += self.beta * prey_weight
                        for idx in sorted(a, reverse=True):
                            del herbivore_class.herbs[pos][idx]
                        break

    def breeding(self, pos):
        """
        breeds animal on the given tile, depending on the set parameters
        :param pos: the position/tile
        :return:
        """
        if pos in self.carns.keys():
            children = []
            n = len(self.carns[pos])
            for idx, animal in enumerate(self.carns[pos]):
                if animal['weight'] < self.zeta * (self.w_birth + self.sigma_birth):
                    p = 0
                else:
                    p = min(1, self.gamma * animal['fitness'] * (n - 1))
                if p > np.random.rand(1):
                    w = np.random.normal(self.w_birth, self.sigma_birth)
                    if animal['weight'] > self.xi * w:
                        children.append({'loc': pos, 'pop': [{'species': 'Carnievore', 'age': 0, 'weight': w}]})
                        self.carns[pos][idx]['weight'] -= self.xi * w
            if len(children) > 0:
                Carnivores.add_carnivores(self, children)

    def migration_calculations(self, rader, kolonner, island_class, herb_class):
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for rad in range(1, rader - 1):
            for kol in range(1, kolonner - 1):
                pos = (rad, kol)
                if pos in self.carns.keys():
                    for idx, animal in enumerate(self.carns[pos]):
                        if animal['fitness'] * self.mu >= np.random.rand(1):
                            if (rad + 1, kol) in self.carns.keys():
                                e_down = herb_class.tot_weight_herbivores((rad + 1, kol)) / (
                                        (len(self.carns[(rad + 1, kol)]) + 1) * self.f)
                            else:
                                e_down = herb_class.tot_weight_herbivores((rad + 1, kol)) / self.f
                            if island_class.fetch_naturetype((rad + 1, kol)) == 'O' or \
                                    island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                                p_down = 0
                            else:
                                p_down = np.exp(self.gamma * e_down)

                            if (rad - 1, kol) in self.carns.keys():
                                e_up = herb_class.tot_weight_herbivores((rad - 1, kol)) / (
                                        (len(self.carns[(rad - 1, kol)]) + 1) * self.f)
                            else:
                                e_up = herb_class.tot_weight_herbivores((rad - 1, kol)) / self.f
                            if island_class.fetch_naturetype((rad - 1, kol)) == 'O' or \
                                    island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                                p_up = 0
                            else:
                                p_up = np.exp(self.gamma * e_up)

                            if (rad, kol - 1) in self.carns.keys():
                                e_left = herb_class.tot_weight_herbivores((rad, kol - 1)) / (
                                            (len(self.carns[(rad, kol - 1)]) + 1) * self.f)
                            else:
                                e_left = herb_class.tot_weight_herbivores((rad, kol - 1)) / self.f
                            if island_class.fetch_naturetype((rad, kol - 1)) == 'O' or \
                                    island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                                p_left = 0
                            else:
                                p_left = np.exp(self.gamma * e_left)

                            if (rad, kol + 1) in self.carns.keys():
                                e_right = herb_class.tot_weight_herbivores((rad, kol + 1)) / (
                                            (len(self.carns[(rad, kol + 1)]) + 1) * self.f)
                            else:
                                e_right = herb_class.tot_weight_herbivores((rad, kol + 1)) / self.f
                            if island_class.fetch_naturetype((rad, kol + 1)) == 'O' or \
                                    island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                                p_right = 0
                            else:
                                p_right = np.exp(self.gamma * e_right)

                            if p_up + p_right + p_left + p_down == 0:
                                break

                            prob_up = p_up / (p_down + p_left + p_right + p_up)
                            prob_down = p_down / (p_down + p_left + p_right + p_up)
                            prob_left = p_left / (p_down + p_left + p_right + p_up)
                            prob_right = p_right / (p_down + p_left + p_right + p_up)

                            direction = np.random.choice(np.arange(1, 5), p=[prob_right, prob_up, prob_left, prob_down])

                            if direction == 1:
                                self.animals_with_new_pos.append({'loc': (rad, kol + 1), 'pop': [animal]})
                            if direction == 2:
                                self.animals_with_new_pos.append({'loc': (rad - 1, kol), 'pop': [animal]})
                            if direction == 3:
                                self.animals_with_new_pos.append({'loc': (rad, kol - 1), 'pop': [animal]})
                            if direction == 4:
                                self.animals_with_new_pos.append({'loc': (rad + 1, kol), 'pop': [animal]})

                            self.idx_for_animals_to_remove.append([pos, idx])

    def migration_execution(self):
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del self.carns[info[0]][info[1]]
        self.add_carnivores(self.animals_with_new_pos)

    def aging(self, pos):
        """
        ages all the animal on one tile with 1 year
        :param pos: the position/tile
        :return:
        """
        if pos in self.carns.keys():
            for idx in range(len(self.carns[pos])):
                self.carns[pos][idx]['age'] += 1

    def loss_of_weight(self, pos):
        """
        Reduces the weight of all the herbivores on a single tile
        :param pos: the position/tile
        :return:
        """
        if pos in self.carns.keys():
            for idx in range(len(self.carns[pos])):
                self.carns[pos][idx]['weight'] -= self.eta * self.carns[pos][idx]['weight']

    def death(self, pos):
        """
        Removes carnivores from the list according to the formula for death
        :param pos: the position asked for
        :return:
        """
        if pos in self.carns.keys():
            a = []
            for idx, animal in enumerate(self.carns[pos]):
                if animal['fitness'] == 0:
                    a.append(idx)
                else:
                    p = self.omega * (1 - animal['fitness'])
                    if p >= np.random.rand(1):
                        a.append(idx)
            for idx in sorted(a, reverse=True):
                del self.carns[pos][idx]






