from rossum_to import *
from rossum import Island

class Cell:
    def __init__(self):
        self.herbs = {}
        self.carns = {}

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


if __name__ == "__main__":
    herbivores = [{'loc': (3, 3), 'pop': [{'species': 'Herbivore', 'age': 20, 'weight': 17.3},
                                          {'species': 'Herbivore', 'age': 30, 'weight': 10.3},
                                          {'species': 'Herbivore', 'age': 10, 'weight': 10.3}]}]
    a = Cell()
    b = Island("OOOOO\nOJJJO\nOJJJO\nOJJJO\nOJJJO\nOOOOO")
    a.add_animal(herbivores, b)
    Herbivores.calculate_fitness(self, pos=(3, 3))
    print(a.herbs((3, 3)))