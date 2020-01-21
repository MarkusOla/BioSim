import numpy as np

class Landscape:
    def set_food(self, pos, food):
        """
        Sets the initial food-values for a tile of a given position
        :param pos: The position of the map
        """
        self.food.update({pos: self.f_max})


    def grow_food(self, pos):
        """
        Updates the amount of food after the animals have eaten in a given position
        :param pos: The position of the map
        """
        self.food.update({pos: self.food[pos] + self.alpha * (self.fsav_max - self.food[pos])})

class Savannah:
    def __init__(self, f_max=300, alpha=0.3):
        self.f_max = f_max
        self.alpha = alpha

class Jungle:
    def __init__(self, f_max=300, alpha=0.3):
        self.f_max = f_max
        self.alpha = alpha

class :
    def __init__(self, f_max=00, alpha=0):
        self.f_max = f_max
        self.alpha = alpha

class Island:
    """ Class to contain information about food and map"""
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

    def set_new_params(self, terrain, new_params):
        """
        Set class parameters.
        Parameters
        ----------
        :param terrain: Which of the naturetype you want to set new parameters for
        :param new_params : dict
            Legal keys: 'fsav_max', 'alpha', 'fjung_max'
        Raises
        ------
        ValueError, KeyError
        """
        default_params = {'S': {'f_max': 300.0,
                                'alpha': 0.3},
                          'J': {'f_max': 800}}

        if terrain in default_params.keys():
            for key in new_params:
                if key not in (default_params[terrain].keys()):
                    raise KeyError('Invalid parameter name: ' + key)
        else:
            raise KeyError('Invalid terrain-type or terrain-type without parameters:' + terrain)

        if terrain == 'S':
            if 'f_max' in new_params:
                if not 0 <= new_params['f_max']:
                    raise ValueError('f_max must be larger or equal to 0')
                self.fsav_max = new_params['f_max']

            if 'alpha' in new_params:
                if 'alpha' in new_params:
                    if not 0 <= new_params['alpha']:
                        raise ValueError('alpha must be larger or equal to 0')
                self.alpha = new_params['alpha']

        if terrain == 'J':
            if 'f_max' in new_params:
                if not 0 <= new_params['f_max']:
                    raise ValueError('f_max must be larger or equal to 0')
                self.fsav_max = new_params['f_max']

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
        """
        Fetches the naturetype of the map in the input position
        :param pos: The given tile
        :return: which naturetype the given tile has
        """
        return self.le_map[pos]

    def food_gets_eaten(self, pos, f_animal):
        """
        reduces the amount of food avaliable on the tiles

        :param f_animal:how much food the herbivore wants to eat
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

    def num_animals(self):
        """
        Goes through each animal list counting the number of ecch species
        :return: the total numer of each species
        """
        num_herbs = 0
        num_carns = 0
        for pos in self.herbs:
            num_herbs += len(self.herbs[pos])
        for pos in self.carns:
            num_carns += len(self.carns[pos])
        return num_herbs, num_carns
