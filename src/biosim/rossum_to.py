import numpy as np
from rossum import Island


class Animal:
    def calculate_fitness(self, pos, animals):
        """
        Calculates the fitness for all the herbivores on one tile
        :param pos: gives which tile we want to calculate the fitness
        :return:
        """
        if pos in animals.keys():
            for animal in animals[pos]:
                if animal['weight'] == 0:
                    new_fitness = {'fitness': 0}
                    animal.update(new_fitness)
                else:
                    new_fitness = {'fitness': (1 / (1 + np.exp(self.phi_age * (animal['age'] - self.a_half)))) *
                                              (1 / (1 + np.exp(-(self.phi_weight * (animal['weight'] - self.w_half)))))}
                    animal.update(new_fitness)

    def sort_by_fitness(self, pos, animals):
        """
        Sorts the herbivores on a tile after their fitness
        :param pos: the position(tile)
        :return:
        """
        if pos in animals.keys():
            animals[pos] = sorted(animals[pos], key=lambda i: i['fitness'], reverse=True)

    def animals_eat(self, pos, island_class, animals):
        """
        herbivores eat, in order of their fitness
        :param pos: the position/tile
        :param island_class: retrives the Island class, to make use of the food_gets_eat function
        :return:
        """
        if pos in animals.keys():

            for idx, animal in enumerate(animals[pos]):
                food = island_class.food_gets_eaten(pos, self.f)
                animals[pos][idx]['weight'] += self.beta * food

    def breeding(self, pos, island_class, animals):
        """
        breeds herbivores on the given tile, depending on the set parameters
        :param pos: the position/tile
        :param island_class: the island, is used as in
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
                if p > np.random.rand(1):
                    w = np.random.normal(self.w_birth, self.sigma_birth)
                    if animal['weight'] > self.xi * w:
                        children.append({'loc': pos, 'pop': [{'species': animal['species'], 'age': 0, 'weight': w}]})
                        animals[pos][idx]['weight'] -= self.xi * w
            if len(children) > 0:
                island_class.add_animals(children)

    def aging(self, pos, animals):
        """
        ages all the herbivores on one tile with 1 year
        :param pos: the position/tile
        :return:
        """
        if pos in animals.keys():
            for idx in range(len(animals[pos])):
                animals[pos][idx]['age'] += 1

    def loss_of_weight(self, pos, animals):
        """
        Reduces the weight of all the herbivores on a single tile
        :param pos: the position/tile
        :return:
        """
        if pos in animals.keys():
            for idx in range(len(animals[pos])):
                animals[pos][idx]['weight'] -= self.eta * animals[pos][idx]['weight']

    def death(self, pos, animals):
        """
        removes herbivores from the list according to the formula for death
        :param pos: the position asked for
        """
        if pos in animals.keys():
            a = []
            for idx, animal in enumerate(animals[pos]):
                if animal['fitness'] == 0:
                    a.append(idx)
                else:
                    p = self.omega * (1 - animal['fitness'])
                    if p >= np.random.rand(1):
                        a.append(idx)
            for idx in sorted(a, reverse=True):
                del animals[pos][idx]


class Herbivores(Animal):
    def __init__(self, w_birth=8.0, sigma_birth=1.5, beta=0.9, eta=0.05, a_half=40.0, phi_age=0.2, w_half=10.0,
                 phi_weight=0.1, mu=0.25, lambda1=1.0, gamma=0.2, zeta=3.5, xi=1.2, omega=0.4, f=10.0):
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
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []

    def set_new_params(self, new_params):
        """
        Set class parameters.
        Parameters
        ----------
        new_params : dict
            Legal keys: 'p_death', 'p_divide'
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
                          'f': 50.0,
                          'DeltaPhiMax': 10.0}

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

        if 'zetta' in new_params:
            if not 0 <= new_params['zetta']:
                raise ValueError('zetta must be larger or equal to 0')
            self.zetta = new_params['zetta']

        if 'xi' in new_params:
            if not 0 <= new_params['xi']:
                raise ValueError('xi must be larger or equal to 0')
            self.xi = new_params['xi']

        if 'omega' in new_params:
            if not 0 <= new_params['omega']:
                raise ValueError('omega must be larger or equal to 0.')
            self.omega = new_params['omega']

        if 'f' in new_params:
            if not 0 <= new_params['f']:
                raise ValueError('f must be larger or equal to 0')
            self.f = new_params['f']

    def sort_before_getting_hunted(self, pos, animals):
        """
        Sorts the herbivores from worst to best fitness
        :param pos:
        :return:
        """
        if pos in animals.keys():
            animals[pos] = sorted(animals[pos], key=lambda i: i['fitness'])

    def migration_calculations(self, rader, kolonner, island_class, animals):
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for rad in range(1, rader - 1):
            for kol in range(1, kolonner - 1):
                pos = (rad, kol)
                if pos in animals.keys():
                    for idx, animal in enumerate(animals[pos]):
                        if animal['fitness'] * self.mu >= np.random.rand(1):
                            if (rad + 1, kol) in animals.keys():
                                e_down = island_class.food[(rad + 1, kol)] / \
                                         ((len(animals[(rad + 1, kol)]) + 1) * self.f)
                            else:
                                e_down = island_class.food[(rad + 1, kol)] / self.f
                            if island_class.fetch_naturetype((rad + 1, kol)) == 'O' \
                                    or island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                                p_down = 0
                            else:
                                p_down = np.exp(self.lambda1 * e_down)

                            if (rad - 1, kol) in animals.keys():
                                e_up = island_class.food[(rad - 1, kol)] / \
                                       ((len(animals[(rad - 1, kol)]) + 1) * self.f)
                            else:
                                e_up = island_class.food[(rad - 1, kol)] / self.f
                            if island_class.fetch_naturetype((rad - 1, kol)) == 'O' \
                                    or island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                                p_up = 0
                            else:
                                p_up = np.exp(self.lambda1 * e_up)

                            if (rad, kol - 1) in animals.keys():
                                e_left = island_class.food[(rad, kol - 1)] / (
                                            (len(animals[(rad, kol - 1)]) + 1) * self.f)
                            else:
                                e_left = island_class.food[(rad, kol - 1)] / self.f
                            if island_class.fetch_naturetype((rad, kol - 1)) == 'O' \
                                    or island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                                p_left = 0
                            else:
                                p_left = np.exp(self.lambda1 * e_left)

                            if (rad, kol + 1) in animals.keys():
                                e_right = island_class.food[(rad, kol + 1)] / (
                                            (len(animals[(rad, kol + 1)]) + 1) * self.f)
                            else:
                                e_right = island_class.food[(rad, kol + 1)] / self.f
                            if island_class.fetch_naturetype((rad, kol + 1)) == 'O' \
                                    or island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                                p_right = 0
                            else:
                                p_right = np.exp(self.lambda1 * e_right)

                            if p_up + p_right + p_left + p_down == 0:
                                break

                            prob_up = p_up / (p_down + p_left + p_right + p_up)
                            prob_down = p_down / (p_down + p_left + p_right + p_up)
                            prob_left = p_left / (p_down + p_left + p_right + p_up)
                            prob_right = p_right / (p_down + p_left + p_right + p_up)

                            direction = np.random.choice(np.arange(1, 5),
                                                         p=[prob_right, prob_up, prob_left, prob_down])

                            if direction == 1:
                                self.animals_with_new_pos.append({'loc': (rad, kol + 1), 'pop': [animal]})
                            if direction == 2:
                                self.animals_with_new_pos.append({'loc': (rad - 1, kol), 'pop': [animal]})
                            if direction == 3:
                                self.animals_with_new_pos.append({'loc': (rad, kol - 1), 'pop': [animal]})
                            if direction == 4:
                                self.animals_with_new_pos.append({'loc': (rad + 1, kol), 'pop': [animal]})

                            self.idx_for_animals_to_remove.append([pos, idx])

    def migration_execution(self, island_class, animals):
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del animals[info[0]][info[1]]
        island_class.add_animals(self.animals_with_new_pos)

    def tot_weight_herbivores(self, pos, animals):
        if pos in animals.keys():
            tot_weight = 0
            for herb in animals[pos]:
                tot_weight += herb['weight']
        else:
            tot_weight = 0
        return tot_weight


class Carnivores(Animal):
    def __init__(self, w_birth=6.0, sigma_birth=1.0, beta=0.75, eta=0.125, a_half=60.0, phi_age=0.4, w_half=4.0,
                 phi_weight=0.4, mu=0.4, lambda1=1.0, gamma=0.8, zeta=3.5, xi=1.1, omega=0.9, f=50.0, DeltaPhiMax=10.0
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
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []

    def set_new_params(self, new_params):
        """
        Set class parameters.
        Parameters
        ----------
        new_params : dict
            Legal keys: 'p_death', 'p_divide'
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
                          'f': 50.0,
                          'DeltaPhiMax': 10.0}

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

        if 'zetta' in new_params:
            if not 0 <= new_params['zetta']:
                raise ValueError('zetta must be larger or equal to 0')
            self.zetta = new_params['zetta']

        if 'xi' in new_params:
            if not 0 <= new_params['xi']:
                raise ValueError('xi must be larger or equal to 0')
            self.xi = new_params['xi']

        if 'omega' in new_params:
            if not 0 <= new_params['omega']:
                raise ValueError('omega must be larger or equal to 0.')
            self.omega = new_params['omega']

        if 'f' in new_params:
            if not 0 <= new_params['f']:
                raise ValueError('f must be larger or equal to 0')
            self.f = new_params['f']

        if 'DeltaPhiMax' in new_params:
            if not 0 < new_params['DeltaPhiMax']:
                raise ValueError('DeltaPhiMax must be larger than 0')
            self.DeltaPhiMax = new_params['DeltaPhiMax']

    def add_carnivores(self, animal_list, animals):
        """
        Adds carnivores to the map according to the input list
        :param animal_list: A list of which animals to put in and where they should be put in
        :return:
        """
        for animal in animal_list:
            if animal['loc'] not in animals.keys():
                animals.update({animal['loc']: animal['pop']})
            else:
                animals[animal['loc']] += animal['pop']

    def carnivores_eat(self, pos, island_class, animals):
        if pos in animals.keys():
            for idx1, carnivore in enumerate(animals[pos]):
                prey_weight = 0
                a = []
                for idx2, herbivore in enumerate(island_class.herbs[pos]):
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
                        animals[pos][idx1]['weight'] += self.beta * self.f
                        for idx in sorted(a, reverse=True):
                            del island_class.herbs[pos][idx]
                        break
                    elif prey_weight > 0 & idx2 == len(island_class.herbs[pos]):
                        animals[pos][idx1]['weight'] += self.beta * prey_weight
                        for idx in sorted(a, reverse=True):
                            del island_class.herbs[pos][idx]
                        break

    def migration_calculations(self, rader, kolonner, island_class, herb_class, animals):
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for rad in range(1, rader - 1):
            for kol in range(1, kolonner - 1):
                pos = (rad, kol)
                if pos in animals.keys():
                    for idx, animal in enumerate(animals[pos]):
                        if animal['fitness'] * self.mu >= np.random.rand(1):
                            if (rad + 1, kol) in animals.keys():
                                e_down = herb_class.tot_weight_herbivores((rad + 1, kol), island_class.herbs) / (
                                        (len(animals[(rad + 1, kol)]) + 1) * self.f)
                            else:
                                e_down = herb_class.tot_weight_herbivores((rad + 1, kol), island_class.herbs) / self.f
                            if island_class.fetch_naturetype((rad + 1, kol)) == 'O' or \
                                    island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                                p_down = 0
                            else:
                                p_down = np.exp(self.lambda1 * e_down)

                            if (rad - 1, kol) in animals.keys():
                                e_up = herb_class.tot_weight_herbivores((rad - 1, kol), island_class.herbs) / (
                                        (len(animals[(rad - 1, kol)]) + 1) * self.f)
                            else:
                                e_up = herb_class.tot_weight_herbivores((rad - 1, kol), island_class.herbs) / self.f
                            if island_class.fetch_naturetype((rad - 1, kol)) == 'O' or \
                                    island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                                p_up = 0
                            else:
                                p_up = np.exp(self.lambda1 * e_up)

                            if (rad, kol - 1) in animals.keys():
                                e_left = herb_class.tot_weight_herbivores((rad, kol - 1), island_class.herbs) / (
                                            (len(animals[(rad, kol - 1)]) + 1) * self.f)
                            else:
                                e_left = herb_class.tot_weight_herbivores((rad, kol - 1), island_class.herbs) / self.f
                            if island_class.fetch_naturetype((rad, kol - 1)) == 'O' or \
                                    island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                                p_left = 0
                            else:
                                p_left = np.exp(self.lambda1 * e_left)

                            if (rad, kol + 1) in animals.keys():
                                e_right = herb_class.tot_weight_herbivores((rad, kol + 1), island_class.herbs) / (
                                            (len(animals[(rad, kol + 1)]) + 1) * self.f)
                            else:
                                e_right = herb_class.tot_weight_herbivores((rad, kol + 1), island_class.herbs) / self.f
                            if island_class.fetch_naturetype((rad, kol + 1)) == 'O' or \
                                    island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                                p_right = 0
                            else:
                                p_right = np.exp(self.lambda1 * e_right)

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

    def migration_execution(self, island_class, animals):
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del animals[info[0]][info[1]]
        island_class.add_animals(self.animals_with_new_pos)


if __name__ == "__main__":
    herbivores = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]}]
    a = Herbivores()
    b = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
    a.add_animal(herbivores, b)
    print(a.animal)
