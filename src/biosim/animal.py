# -*- coding: utf-8 -*-

"""
This code will contain the island class
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

import numpy as np
import random
import math


class Animal:
    """
    Class for animal functions
    """
    def calculate_fitness(self, pos, animals):
        """
        Calculates the fitness for all the animals on one tile.

        :param pos: Gives the position were we want to calculate the fitness
        :param animals: The animal dictionary for the given species
        """
        if pos in animals.keys():
            for animal in animals[pos]:
                if animal['weight'] == 0:
                    new_fitness = {'fitness': 0}
                    animal.update(new_fitness)
                else:
                    new_fitness = {'fitness': (1 / (1 + math.exp(self.phi_age * (animal['age'] - self.a_half)))) *
                                              (1 / (1 + math.exp(-(self.phi_weight * (animal['weight']
                                                                                      - self.w_half)))))}
                    animal.update(new_fitness)

    @staticmethod
    def sort_by_fitness(pos, animals):
        """
        Sorts the animals after their fitness, best to worst.

        :param pos: the position(tile)
        :param animals: the animal dictionary
        """
        if pos in animals.keys():
            animals[pos] = sorted(animals[pos], key=lambda i: i['fitness'], reverse=True)

    def breeding(self, pos, island_class, animals):
        """
        Breeds animals on the given position for the given species.

        :param pos: The position/tile
        :param island_class: The island-class, used to add animals
        :param animals: the dictionary contain info about all the selected species
        :return:
        """
        if pos in animals.keys():
            children = []
            n = len(animals[pos])
            for idx, animal in enumerate(animals[pos]):
                if animal['weight'] < self.zeta * (self.w_birth + self.sigma_birth):
                    p = 0
                else:
                    p = min(1, self.gamma * animal['fitness'] * (n - 1))
                if p > random.random():
                    w = np.random.normal(self.w_birth, self.sigma_birth)
                    if animal['weight'] > self.xi * w:
                        children.append({'loc': pos, 'pop': [{'species': animal['species'], 'age': 0, 'weight': w}]})
                        animals[pos][idx]['weight'] -= self.xi * w
            if len(children) > 0:
                island_class.add_animals(children)

    @staticmethod
    def aging(pos, animals):
        """
        Ages all the animals on one tile with 1 year.

        :param pos: The position/tile
        :param animals: The animal dictionary
        """
        if pos in animals.keys():
            for idx in range(len(animals[pos])):
                animals[pos][idx]['age'] += 1

    def loss_of_weight(self, pos, animals):
        """
        Reduces the weight of all the animals on a single tile

        :param pos: The position/tile
        :param animals: The animal dictionary
        """
        if pos in animals.keys():
            for idx in range(len(animals[pos])):
                animals[pos][idx]['weight'] -= self.eta * animals[pos][idx]['weight']

    def death(self, pos, animals):
        """
        Removes  from the animals list according to the formula for death
        :param pos: the position asked for
        :param animals: animal-dictionary
        """
        if pos in animals.keys():
            a = []
            for idx, animal in enumerate(animals[pos]):
                if animal['fitness'] == 0:
                    a.append(idx)
                else:
                    p = self.omega * (1 - animal['fitness'])
                    if p >= random.random():
                        a.append(idx)
            for idx in sorted(a, reverse=True):
                del animals[pos][idx]


class Herbivores(Animal):
    """
    The animal subclass for Herbivores
    """
    def __init__(self, w_birth=8.0, sigma_birth=1.5, beta=0.9, eta=0.05, a_half=40.0, phi_age=0.2, w_half=10.0,
                 phi_weight=0.1, mu=0.25, lambda1=1.0, gamma=0.2, zeta=3.5, xi=1.2, omega=0.4, f=10.0):
        """
        The class containing all the necessary functions for herbivores.

        :param w_birth: The average weight for a newborn Herbivore
        :param sigma_birth: The standard deviation for weight of a newborn
        :param beta: The growing factor telling how much of the food is changed into weight
        :param eta: The weight reduction factor
        :param a_half: Fitness-factor
        :param phi_age: Fitness-factor
        :param w_half: Fitness-factor
        :param phi_weight: Fitness-factor
        :param mu: Factor used to calculate probability for migration
        :param lambda1: Migration-factor
        :param gamma: Gives the probability for giving birth, given number of animals on same tiles and their fitness
        :param zeta: Gives the restrictions for giving girth depending on weight
        :param xi: The factor for weight loss after given birth
        :param omega: The probability of dieing given the animals fitnessvalue
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
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []

    def set_new_params(self, new_params):
        """
        Set class parameters.
        Parameters
        ----------
        new_params : dict
            Legal keys: 'w_birth', 'sigma_birth', 'beta', 'eta', 'a_half', 'phi_age', 'w_half', 'phi_weight', 'mu',
                        'lambda', 'gamma', 'zeta', 'xi', 'omega', 'F'
        Raises
        ------
        ValueError, KeyError
        """
        default_params = {'w_birth': 8.0,
                          'sigma_birth': 1.5,
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
                          'F': 10.0}

        if type(new_params) is not dict:
            raise TypeError('new_params is not a dictionary')

        for key in new_params:
            if key not in (default_params.keys()):
                raise KeyError('Invalid parameter name: ' + key)

        if 'w_birth' in new_params:
            if not 0 <= new_params['w_birth']:
                raise ValueError('birth_weight must be larger or equal to 0')
            self.w_birth = new_params['w_birth']

        if 'sigma_birth' in new_params:
            if not 0 <= new_params['sigma_birth']:
                raise ValueError('sigma_birth must be larger or equal to 0')
            self.sigma_birth = new_params['sigma_birth']

        if 'beta' in new_params:
            if not 0 <= new_params['beta']:
                raise ValueError('p_death must be larger or equal to 0.')
            self.beta = new_params['beta']

        if 'eta' in new_params:
            if not 0 <= new_params['eta'] <= 1:
                raise ValueError('p_divide must be in [0, 1].')
            self.eta = new_params['eta']

        if 'a_half' in new_params:
            if not 0 <= new_params['a_half']:
                raise ValueError('a_half must be larger or equal to 0.')
            self.a_half = new_params['a_half']

        if 'phi_age' in new_params:
            if not 0 <= new_params['phi_age']:
                raise ValueError('phi_age must be larger or equal to 0.')
            self.phi_age = new_params['phi_age']

        if 'w_half' in new_params:
            if not 0 <= new_params['w_half']:
                raise ValueError('w_half must be larger or equal to 0')
            self.w_half = new_params['w_half']

        if 'phi_weight' in new_params:
            if not 0 <= new_params['phi_weight']:
                raise ValueError('phi_weight must be larger or equal to 0.')
            self.phi_weight = new_params['phi_weight']

        if 'mu' in new_params:
            if not 0 <= new_params['mu']:
                raise ValueError('mu must be larger or equal to 0.')
            self.mu = new_params['mu']

        if 'lambda' in new_params:
            if not 0 <= new_params['lambda']:
                raise ValueError('lambda must be larger or equal to 0.')
            self.lambda1 = new_params['lambda']

        if 'gamma' in new_params:
            if not 0 <= new_params['gamma']:
                raise ValueError('gamma must be larger or equal to 0.')
            self.gamma = new_params['gamma']

        if 'zeta' in new_params:
            if not 0 <= new_params['zeta']:
                raise ValueError('zetta must be larger or equal to 0')
            self.zeta = new_params['zeta']

        if 'xi' in new_params:
            if not 0 <= new_params['xi']:
                raise ValueError('xi must be larger or equal to 0')
            self.xi = new_params['xi']

        if 'omega' in new_params:
            if not 0 <= new_params['omega']:
                raise ValueError('omega must be larger or equal to 0.')
            self.omega = new_params['omega']

        if 'F' in new_params:
            if not 0 <= new_params['F']:
                raise ValueError('F must be larger or equal to 0')
            self.f = new_params['F']

    @staticmethod
    def sort_before_getting_hunted(pos, animals):
        """
        Sorts the herbivores from worst to best fitness.

        :param pos: The position to sort the animals
        :param animals: The animal dictionary that shall be sorted
        """
        if pos in animals.keys():
            animals[pos] = sorted(animals[pos], key=lambda i: i['fitness'])

    def migration_calculations(self, island_class, animals):
        """
        Calculates which animals shall move and where they shall move to, makes one list with animals that will be
        placed into new positions in the migration_execution-function and one list with animals that are deleted
        in the migration_execution-function.

        :param island_class: The Island-class used to check terrain-type and amount of food in the neighbour tiles
        :param animals: The herbivore dictionary
        """
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for pos in animals.keys():
            rad = pos[0]
            kol = pos[1]
            for idx, animal in enumerate(animals[pos]):
                if animal['fitness'] * self.mu >= random.random():
                    if (rad + 1, kol) in animals.keys():
                        e_down = island_class.food[(rad + 1, kol)] / \
                                 ((len(animals[(rad + 1, kol)]) + 1) * self.f)
                    else:
                        e_down = island_class.food[(rad + 1, kol)] / self.f
                    if island_class.fetch_naturetype((rad + 1, kol)) == 'O' \
                            or island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                        p_down = 0
                    else:
                        p_down = math.exp(self.lambda1 * e_down)

                    if (rad - 1, kol) in animals.keys():
                        e_up = island_class.food[(rad - 1, kol)] / \
                               ((len(animals[(rad - 1, kol)]) + 1) * self.f)
                    else:
                        e_up = island_class.food[(rad - 1, kol)] / self.f
                    if island_class.fetch_naturetype((rad - 1, kol)) == 'O' \
                            or island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                        p_up = 0
                    else:
                        p_up = math.exp(self.lambda1 * e_up)

                    if (rad, kol - 1) in animals.keys():
                        e_left = island_class.food[(rad, kol - 1)] / (
                                    (len(animals[(rad, kol - 1)]) + 1) * self.f)
                    else:
                        e_left = island_class.food[(rad, kol - 1)] / self.f
                    if island_class.fetch_naturetype((rad, kol - 1)) == 'O' \
                            or island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                        p_left = 0
                    else:
                        p_left = math.exp(self.lambda1 * e_left)

                    if (rad, kol + 1) in animals.keys():
                        e_right = island_class.food[(rad, kol + 1)] / (
                                    (len(animals[(rad, kol + 1)]) + 1) * self.f)
                    else:
                        e_right = island_class.food[(rad, kol + 1)] / self.f
                    if island_class.fetch_naturetype((rad, kol + 1)) == 'O' \
                            or island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                        p_right = 0
                    else:
                        p_right = math.exp(self.lambda1 * e_right)

                    if p_up + p_right + p_left + p_down == 0:
                        break

                    prob_up = p_up / (p_down + p_left + p_right + p_up)
                    prob_down = p_down / (p_down + p_left + p_right + p_up)
                    prob_right = p_right / (p_down + p_left + p_right + p_up)

                    direction = random.random()

                    if direction <= prob_right:
                        self.animals_with_new_pos.append({'loc': (rad, kol + 1), 'pop': [animal]})
                    elif prob_right < direction <= (prob_right + prob_up):
                        self.animals_with_new_pos.append({'loc': (rad - 1, kol), 'pop': [animal]})
                    elif (prob_right + prob_up) < direction <= (1 - prob_down):
                        self.animals_with_new_pos.append({'loc': (rad, kol - 1), 'pop': [animal]})
                    else:
                        self.animals_with_new_pos.append({'loc': (rad + 1, kol), 'pop': [animal]})

                    self.idx_for_animals_to_remove.append([pos, idx])

    def migration_execution(self, island_class, animals):
        """
        Function that executes what is being calculated in the migration_calculations-functions, removes the animals
        that are migrating from their current positions and places them in their new position.

        :param island_class: The Island-class used to add animals to new positions
        :param animals: The herbivore dictionary
        """
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del animals[info[0]][info[1]]
        island_class.add_animals(self.animals_with_new_pos)

    def animals_eat(self, pos, island_class, animals):
        """
        Herbivores eat, in order of their fitness.

        :param pos: The position/tile
        :param island_class: Retrives the Island-class, to make use of the food_gets_eat function
        :param animals: The herbivore dictionary
        """
        if pos in animals.keys():

            for idx, animal in enumerate(animals[pos]):
                food = island_class.food_gets_eaten(pos, self.f)
                animals[pos][idx]['weight'] += self.beta * food

    @staticmethod
    def tot_weight_herbivores(pos, animals):
        """
        Function to calculate the total weight of the herbivores, it is used in the carnivore migration_calculation-
        function.

        :param pos: The position for which we calculate the total weight
        :param animals: The animal dictionary, in this case the dictionary for herbivores
        """
        if pos in animals.keys():
            tot_weight = 0
            for herb in animals[pos]:
                tot_weight += herb['weight']
        else:
            tot_weight = 0
        return tot_weight


class Carnivores(Animal):
    def __init__(self, w_birth=6.0, sigma_birth=1.0, beta=0.75, eta=0.125, a_half=60.0, phi_age=0.4, w_half=4.0,
                 phi_weight=0.4, mu=0.4, lambda1=1.0, gamma=0.8, zeta=3.5, xi=1.1, omega=0.9, f=50.0, deltaphimax=10.0
                 ):
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
        :param mu: Probability for moving
        :param lambda1: Migration-factor
        :param gamma: gives the probability for giving birth, given number of animals on same tiles and their fitness
        :param zeta: Gives the restrictions for giving girth depending on weight
        :param xi: The factor for weight loss after given birth
        :param omega: The probability of dying given the animals fitness-value
        :param f: The maximum value that the carnivores eat
        :param deltaphimax: Constant used to calculate if the carnivore eats the herbivore
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
        self.deltaphimax = deltaphimax
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []

    def set_new_params(self, new_params):
        """
        Set class parameters.
        Parameters
        ----------
        new_params : dict
            Legal keys: 'w_birth', 'sigma_birth', 'beta', 'eta', 'a_half', 'phi_age', 'w_half', 'phi_weight', 'mu',
                        'lambda', 'gamma', 'zeta', 'xi', 'omega', 'F'
        Raises
        ------
        ValueError, KeyError
        """
        default_params = {'w_birth': 6.0,
                          'sigma_birth': 1.0,
                          'beta': 0.7,
                          'eta': 0.125,
                          'a_half': 60.0,
                          'phi_age': 0.4,
                          'w_half': 4.0,
                          'phi_weight': 0.4,
                          'mu': 0.4,
                          'lambda': 1.0,
                          'gamma': 0.8,
                          'zeta': 3.5,
                          'xi': 1.1,
                          'omega': 0.9,
                          'F': 50.0,
                          'DeltaPhiMax': 10.0}

        if type(new_params) is not dict:
            raise TypeError('new_params is not a dictionary')

        for key in new_params:
            if key not in (default_params.keys()):
                raise KeyError('Invalid parameter name: ' + key)

        if 'w_birth' in new_params:
            if not 0 <= new_params['w_birth']:
                raise ValueError('birth_weight must be larger or equal to 0')
            self.w_birth = new_params['w_birth']

        if 'sigma_birth' in new_params:
            if not 0 <= new_params['sigma_birth']:
                raise ValueError('sigma_birth must be larger or equal to 0')
            self.sigma_birth = new_params['sigma_birth']

        if 'beta' in new_params:
            if not 0 <= new_params['beta']:
                raise ValueError('p_death must be larger or equal to 0.')
            self.beta = new_params['beta']

        if 'eta' in new_params:
            if not 0 <= new_params['eta'] <= 1:
                raise ValueError('p_divide must be in [0, 1].')
            self.eta = new_params['eta']

        if 'a_half' in new_params:
            if not 0 <= new_params['a_half']:
                raise ValueError('a_half must be larger or equal to 0.')
            self.a_half = new_params['a_half']

        if 'phi_age' in new_params:
            if not 0 <= new_params['phi_age']:
                raise ValueError('phi_age must be larger or equal to 0.')
            self.phi_age = new_params['phi_age']

        if 'w_half' in new_params:
            if not 0 <= new_params['w_half']:
                raise ValueError('w_half must be larger or equal to 0')
            self.w_half = new_params['w_half']

        if 'phi_weight' in new_params:
            if not 0 <= new_params['phi_weight']:
                raise ValueError('phi_weight must be larger or equal to 0.')
            self.phi_weight = new_params['phi_weight']

        if 'mu' in new_params:
            if not 0 <= new_params['mu']:
                raise ValueError('mu must be larger or equal to 0.')
            self.mu = new_params['mu']

        if 'lambda' in new_params:
            if not 0 <= new_params['lambda']:
                raise ValueError('lambda must be larger or equal to 0.')
            self.lambda1 = new_params['lambda']

        if 'gamma' in new_params:
            if not 0 <= new_params['gamma']:
                raise ValueError('gamma must be larger or equal to 0.')
            self.gamma = new_params['gamma']

        if 'zeta' in new_params:
            if not 0 <= new_params['zeta']:
                raise ValueError('zeta must be larger or equal to 0')
            self.zeta = new_params['zeta']

        if 'xi' in new_params:
            if not 0 <= new_params['xi']:
                raise ValueError('xi must be larger or equal to 0')
            self.xi = new_params['xi']

        if 'omega' in new_params:
            if not 0 <= new_params['omega']:
                raise ValueError('omega must be larger or equal to 0.')
            self.omega = new_params['omega']

        if 'F' in new_params:
            if not 0 <= new_params['F']:
                raise ValueError('F must be larger or equal to 0')
            self.f = new_params['F']

        if 'DeltaPhiMax' in new_params:
            if not 0 < new_params['DeltaPhiMax']:
                raise ValueError('DeltaPhiMax must be larger than 0')
            self.deltaphimax = new_params['DeltaPhiMax']

    def carnivores_eat(self, pos, island_class, animals):
        """
        Function for the carnivores to eat.

        :param pos: Position for which the animals shall eat
        :param island_class: The Island-class, used to import the herbivores
        :param animals: The carnivore dictionary
        """
        if pos in animals.keys():
            for idx1, carnivore in enumerate(animals[pos]):
                prey_weight = 0
                a = []
                if pos in island_class.herbs.keys():
                    for idx2, herbivore in enumerate(island_class.herbs[pos]):
                        if carnivore['fitness'] <= herbivore['fitness']:
                            p = 0
                        elif carnivore['fitness'] - herbivore['fitness'] < self.deltaphimax:
                            p = (carnivore['fitness'] - herbivore['fitness']) / self.deltaphimax
                        else:
                            p = 1
                        if p > random.random():
                            prey_weight += herbivore['weight']
                            a.append(idx2)
                        if prey_weight > self.f:
                            animals[pos][idx1]['weight'] += self.beta * self.f
                            for idx in sorted(a, reverse=True):
                                del island_class.herbs[pos][idx]
                            break
                        elif idx2 == len(island_class.herbs[pos]) - 1:
                            animals[pos][idx1]['weight'] += self.beta * prey_weight
                            for idx in sorted(a, reverse=True):
                                del island_class.herbs[pos][idx]
                            break

    def migration_calculations(self, island_class, herb_class, animals):
        """
        Calculates which animals shall move and where they shall move to, makes one list with animals that will be
        placed into new positions in the migration_execution-function and one list with animals that are deleted
        in the migration_execution-function.
        :param island_class: The Island-class used to check terrain-type and total weight of the herbivores
                             in the neighbour tiles
        :param herb_class: The Herbivore-class imported to use calculate total weight of the herbivores
        :param animals: The carnivore dictionary
        """
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for pos in animals.keys():
            rad = pos[0]
            kol = pos[1]
            for idx, animal in enumerate(animals[pos]):
                if animal['fitness'] * self.mu >= random.random():
                    if (rad + 1, kol) in animals.keys():
                        e_down = herb_class.tot_weight_herbivores((rad + 1, kol), island_class.herbs) / (
                                (len(animals[(rad + 1, kol)]) + 1) * self.f)
                    else:
                        e_down = herb_class.tot_weight_herbivores((rad + 1, kol), island_class.herbs) / self.f
                    if island_class.fetch_naturetype((rad + 1, kol)) == 'O' or \
                            island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                        p_down = 0
                    else:
                        p_down = math.exp(self.lambda1 * e_down)

                    if (rad - 1, kol) in animals.keys():
                        e_up = herb_class.tot_weight_herbivores((rad - 1, kol), island_class.herbs) / (
                                (len(animals[(rad - 1, kol)]) + 1) * self.f)
                    else:
                        e_up = herb_class.tot_weight_herbivores((rad - 1, kol), island_class.herbs) / self.f
                    if island_class.fetch_naturetype((rad - 1, kol)) == 'O' or \
                            island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                        p_up = 0
                    else:
                        p_up = math.exp(self.lambda1 * e_up)

                    if (rad, kol - 1) in animals.keys():
                        e_left = herb_class.tot_weight_herbivores((rad, kol - 1), island_class.herbs) / (
                                    (len(animals[(rad, kol - 1)]) + 1) * self.f)
                    else:
                        e_left = herb_class.tot_weight_herbivores((rad, kol - 1), island_class.herbs) / self.f
                    if island_class.fetch_naturetype((rad, kol - 1)) == 'O' or \
                            island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                        p_left = 0
                    else:
                        p_left = math.exp(self.lambda1 * e_left)

                    if (rad, kol + 1) in animals.keys():
                        e_right = herb_class.tot_weight_herbivores((rad, kol + 1), island_class.herbs) / (
                                    (len(animals[(rad, kol + 1)]) + 1) * self.f)
                    else:
                        e_right = herb_class.tot_weight_herbivores((rad, kol + 1), island_class.herbs) / self.f
                    if island_class.fetch_naturetype((rad, kol + 1)) == 'O' or \
                            island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                        p_right = 0
                    else:
                        p_right = math.exp(self.lambda1 * e_right)

                    if p_up + p_right + p_left + p_down == 0:
                        break

                    prob_up = p_up / (p_down + p_left + p_right + p_up)
                    prob_down = p_down / (p_down + p_left + p_right + p_up)
                    prob_right = p_right / (p_down + p_left + p_right + p_up)

                    direction = random.random()

                    if direction <= prob_right:
                        self.animals_with_new_pos.append({'loc': (rad, kol + 1), 'pop': [animal]})
                    elif prob_right < direction <= (prob_right + prob_up):
                        self.animals_with_new_pos.append({'loc': (rad - 1, kol), 'pop': [animal]})
                    elif (prob_right + prob_up) < direction <= (1 - prob_down):
                        self.animals_with_new_pos.append({'loc': (rad, kol - 1), 'pop': [animal]})
                    else:
                        self.animals_with_new_pos.append({'loc': (rad + 1, kol), 'pop': [animal]})

                    self.idx_for_animals_to_remove.append([pos, idx])

    def migration_execution(self, island_class, animals):
        """
        Function that executes what is being calculated in the migration_calculations-functions, removes the animals
        that are migrating from their current positions and places them in their new position.

        :param island_class: The Island-class used to add animals to new positions
        :param animals: The carnivore dictionary
        """
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del animals[info[0]][info[1]]
        island_class.add_animals(self.animals_with_new_pos)
