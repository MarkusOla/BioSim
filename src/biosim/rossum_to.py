import numpy as np
from rossum import Island, Fodder


class Animal:
    params = None
    def calculate_fitness(self, pos):
        """
        Calculates the fitness for all the herbivores on one tile
        :param pos: gives which tile we want to calculate the fitness
        :return:
        """
        if pos in self.animal.keys():
            for animal in self.animal[pos]:
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
        if pos in self.animal.keys():
            self.animal[pos] = sorted(self.animal[pos], key=lambda i: i['fitness'], reverse=True)

    def animals_eat(self, pos, food_class):
        """
        herbivores eat, in order of their fitness
        :param pos: the position/tile
        :param food_class: retrives the fodder class, to make use of the food_gets_eat function
        :return:
        """
        if pos in self.animal.keys():

            for idx, animal in enumerate(self.animal[pos]):
                food = food_class.food_gets_eaten(pos)
                self.animal[pos][idx]['weight'] += self.beta * food

    def breeding(self, pos, island_class):
        """
        breeds herbivores on the given tile, depending on the set parameters
        :param pos: the position/tile
        :param island_class: the island, is used as in
        :return:
        """
        if pos in self.animal.keys():
            children = []
            n = len(self.animal[pos])
            for idx, animal in enumerate(self.animal[pos]):
                if animal['weight'] < self.zeta * (self.w_birth + self.sigma_birth):
                    p = 0
                else:
                    p = min(1, self.gamma * animal['fitness'] * (n - 1))
                if p > np.random.rand(1):
                    w = np.random.normal(self.w_birth, self.sigma_birth)
                    if animal['weight'] > self.xi * w:
                        children.append({'loc': pos, 'pop': [{'species': 'Herbievore', 'age': 0, 'weight': w}]})
                        self.animal[pos][idx]['weight'] -= self.xi * w
            if len(children) > 0:
                Herbivores.add_animal(self, children, island_class)

    def aging(self, pos):
        """
        ages all the herbivores on one tile with 1 year
        :param pos: the position/tile
        :return:
        """
        if pos in self.animal.keys():
            for idx in range(len(self.animal[pos])):
                self.animal[pos][idx]['age'] += 1

    def loss_of_weight(self, pos):
        """
        Reduces the weight of all the herbivores on a single tile
        :param pos: the position/tile
        :return:
        """
        if pos in self.animal.keys():
            for idx in range(len(self.animal[pos])):
                self.animal[pos][idx]['weight'] -= self.eta * self.animal[pos][idx]['weight']

    def death(self, pos):
        """
        removes herbivores from the list according to the formula for death
        :param pos: the position asked for
        """
        if pos in self.animal.keys():
            a = []
            for idx, animal in enumerate(self.animal[pos]):
                if animal['fitness'] == 0:
                    a.append(idx)
                else:
                    p = self.omega * (1 - animal['fitness'])
                    if p >= np.random.rand(1):
                        a.append(idx)
            for idx in sorted(a, reverse=True):
                del self.animal[pos][idx]


class Herbivores(Animal):
    def __init__(self, w_birth=8.0, sigma_birth=1.5, beta=0.9, eta=0.05, a_half=40.0, phi_age=0.2, w_half=10.0,
                 phi_weight=0.1, mu=0.25, lambda1=1.0, gamma=0.2, zeta=3.5, xi=1.2, omega=0.4, seed=1, f=10.0):
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
        self.animal = {}
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
            if animal['loc'] not in self.animal.keys():
                self.animal.update({animal['loc']: animal['pop']})
            else:
                self.animal[animal['loc']] += animal['pop']

    def sort_before_getting_hunted(self, pos):
        """
        Sorts the herbivores from worst to best fitness
        :param pos:
        :return:
        """
        if pos in self.animal.keys():
            self.animal[pos] = sorted(self.animal[pos], key=lambda i: i['fitness'])

    def migration_calculations(self, rader, kolonner, island_class, food_class):
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for rad in range(1, rader - 1):
            for kol in range(1, kolonner - 1):
                pos = (rad, kol)
                if pos in self.animal.keys():
                    for idx, animal in enumerate(self.animal[pos]):
                        if animal['fitness'] * self.mu >= np.random.rand(1):
                            if (rad + 1, kol) in self.animal.keys():
                                e_down = food_class.food[(rad + 1, kol)] / \
                                         ((len(self.animal[(rad + 1, kol)]) + 1) * self.f)
                            else:
                                e_down = food_class.food[(rad + 1, kol)] / self.f
                            if island_class.fetch_naturetype((rad + 1, kol)) == 'O' \
                                    or island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                                p_down = 0
                            else:
                                p_down = np.exp(self.gamma * e_down)

                            if (rad - 1, kol) in self.animal.keys():
                                e_up = food_class.food[(rad - 1, kol)] / \
                                       ((len(self.animal[(rad - 1, kol)]) + 1) * self.f)
                            else:
                                e_up = food_class.food[(rad - 1, kol)] / self.f
                            if island_class.fetch_naturetype((rad - 1, kol)) == 'O' \
                                    or island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                                p_up = 0
                            else:
                                p_up = np.exp(self.gamma * e_up)

                            if (rad, kol - 1) in self.animal.keys():
                                e_left = food_class.food[(rad, kol - 1)] / (
                                            (len(self.animal[(rad, kol - 1)]) + 1) * self.f)
                            else:
                                e_left = food_class.food[(rad, kol - 1)] / self.f
                            if island_class.fetch_naturetype((rad, kol - 1)) == 'O' \
                                    or island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                                p_left = 0
                            else:
                                p_left = np.exp(self.gamma * e_left)

                            if (rad, kol + 1) in self.animal.keys():
                                e_right = food_class.food[(rad, kol + 1)] / (
                                            (len(self.animal[(rad, kol + 1)]) + 1) * self.f)
                            else:
                                e_right = food_class.food[(rad, kol + 1)] / self.f
                            if island_class.fetch_naturetype((rad, kol + 1)) == 'O' \
                                    or island_class.fetch_naturetype((rad, kol + 1)) == 'M':
                                p_right = 0
                            else:
                                p_right = np.exp(self.gamma * e_right)

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

    def migration_execution(self, island_class):
        for info in sorted(self.idx_for_animals_to_remove, reverse=True):
            del self.animal[info[0]][info[1]]
        self.add_animal(self.animals_with_new_pos, island_class)

    def tot_weight_herbivores(self, pos):
        if pos in self.animal.keys():
            tot_weight = 0
            for herb in self.animal[pos]:
                tot_weight += herb['weight']
        else:
            tot_weight = 0
        return tot_weight


class Carnivores(Animal):
    #    parameters = {}

    #   @classmethod
    #    def set_parameters(cls):
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
        self.animal = {}
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
            if animal['loc'] not in self.animal.keys():
                self.animal.update({animal['loc']: animal['pop']})
            else:
                self.animal[animal['loc']] += animal['pop']

    def carnivores_eat(self, pos, herbivore_class):
        if pos in self.animal.keys():
            for idx1, carnivore in enumerate(self.animal[pos]):
                prey_weight = 0
                a = []
                for idx2, herbivore in enumerate(herbivore_class.animal[pos]):
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
                        self.animal[pos][idx1]['weight'] += self.beta * self.f
                        for idx in sorted(a, reverse=True):
                            del herbivore_class.animal[pos][idx]
                        break
                    elif prey_weight > 0 & idx2 == len(herbivore_class.animal[pos]):
                        self.animal[pos][idx1]['weight'] += self.beta * prey_weight
                        for idx in sorted(a, reverse=True):
                            del herbivore_class.animal[pos][idx]
                        break

    def migration_calculations(self, rader, kolonner, island_class, herb_class):
        self.animals_with_new_pos = []
        self.idx_for_animals_to_remove = []
        for rad in range(1, rader - 1):
            for kol in range(1, kolonner - 1):
                pos = (rad, kol)
                if pos in self.animal.keys():
                    for idx, animal in enumerate(self.animal[pos]):
                        if animal['fitness'] * self.mu >= np.random.rand(1):
                            if (rad + 1, kol) in self.animal.keys():
                                e_down = herb_class.tot_weight_herbivores((rad + 1, kol)) / (
                                        (len(self.animal[(rad + 1, kol)]) + 1) * self.f)
                            else:
                                e_down = herb_class.tot_weight_herbivores((rad + 1, kol)) / self.f
                            if island_class.fetch_naturetype((rad + 1, kol)) == 'O' or \
                                    island_class.fetch_naturetype((rad + 1, kol)) == 'M':
                                p_down = 0
                            else:
                                p_down = np.exp(self.gamma * e_down)

                            if (rad - 1, kol) in self.animal.keys():
                                e_up = herb_class.tot_weight_herbivores((rad - 1, kol)) / (
                                        (len(self.animal[(rad - 1, kol)]) + 1) * self.f)
                            else:
                                e_up = herb_class.tot_weight_herbivores((rad - 1, kol)) / self.f
                            if island_class.fetch_naturetype((rad - 1, kol)) == 'O' or \
                                    island_class.fetch_naturetype((rad - 1, kol)) == 'M':
                                p_up = 0
                            else:
                                p_up = np.exp(self.gamma * e_up)

                            if (rad, kol - 1) in self.animal.keys():
                                e_left = herb_class.tot_weight_herbivores((rad, kol - 1)) / (
                                            (len(self.animal[(rad, kol - 1)]) + 1) * self.f)
                            else:
                                e_left = herb_class.tot_weight_herbivores((rad, kol - 1)) / self.f
                            if island_class.fetch_naturetype((rad, kol - 1)) == 'O' or \
                                    island_class.fetch_naturetype((rad, kol - 1)) == 'M':
                                p_left = 0
                            else:
                                p_left = np.exp(self.gamma * e_left)

                            if (rad, kol + 1) in self.animal.keys():
                                e_right = herb_class.tot_weight_herbivores((rad, kol + 1)) / (
                                            (len(self.animal[(rad, kol + 1)]) + 1) * self.f)
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
            del self.animal[info[0]][info[1]]
        self.add_carnivores(self.animals_with_new_pos)


if __name__ == "__main__":
    herbivores = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]}]
    a = Herbivores()
    b = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
    a.add_animal(herbivores, b)
    print(a.animal)
