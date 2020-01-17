from rossum_to import Animal, Herbivores, Carnivores
from rossum import Island

class Cell:
    def __init__(self):
        self.herbs = {}
        self.carns = {}

    def add_animals(self, animal_list, island_class):
        """
       Adds herbivore to the map
        :param animal_list: A list that contains the animals wegiht, age and species and where we want to add them
        :return:
       """
        for dict in animal_list:
            for animal in dict['pop']:
                if island_class.fetch_naturetype(dict['loc']) == 'O' or \
                        island_class.fetch_naturetype(dict['loc']) == 'M':
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