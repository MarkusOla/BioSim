# -*- coding: utf-8 -*-

"""
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"


from biosim.rossum import Island, Fodder, Herbivores


class BioSim:
    def __init__(
        self,
        island_map,
        ini_pop,
        seed,
        ymax_animals=None,
        cmax_animals=None,
        img_base=None,
        img_fmt="png",
    ):
        """
        :param island_map: Multi-line string specifying island geography
        :param ini_pop: List of dictionaries specifying initial population
        :param seed: Integer used as random number seed
        :param ymax_animals: Number specifying y-axis limit for graph showing animal numbers
        :param cmax_animals: Dict specifying color-code limits for animal densities
        :param img_base: String with beginning of file name for figures, including path
        :param img_fmt: String with file type for figures, e.g. 'png'

        If ymax_animals is None, the y-axis limit should be adjusted automatically.

        If cmax_animals is None, sensible, fixed default values should be used.
        cmax_animals is a dict mapping species names to numbers, e.g.,
           {'Herbivore': 50, 'Carnivore': 20}

        If img_base is None, no figures are written to file.
        Filenames are formed as

            '{}_{:05d}.{}'.format(img_base, img_no, img_fmt)

        where img_no are consecutive image numbers starting from 0.
        img_base should contain a path and beginning of a file name.
        """
        self.ini_pop = ini_pop
        self.seed = seed
        self.ymax_animals = ymax_animals
        self.cmax_animals = cmax_animals
        self.img_base = img_base
        self.img_fmt = img_fmt
        self.island = Island(island_map)
        self.island.limit_map_vals()
        self.herbivores = Herbivores(seed=seed)
        self.food = Fodder()

    def set_animal_parameters(self, species, params):
        """
        Set parameters for animal species.

        :param species: String, name of animal species
        :param params: Dict with valid parameter specification for species
        """

    def set_landscape_parameters(self, landscape, params):
        """
        Set parameters for landscape type.

        :param landscape: String, code letter for landscape
        :param params: Dict with valid parameter specification for landscape
        """

    def setup_simulation(self):
        for i in range(self.island.rader):
            for j in range(self.island.col):
                self.food.set_food((i, j), self.island)
        self.herbivores.add_animal(self.ini_pop)

    def simulate(self, num_years, vis_years=1, img_years=None):
        """
        Run simulation while visualizing the result.

        :param num_years: number of years to simulate
        :param vis_years: years between visualization updates
        :param img_years: years between visualizations saved to files (default: vis_years)

        Image files will be numbered consecutively.
        """
        for year in range(num_years):
            for i in range(self.island.rader):
                for j in range(self.island.col):
                    pos = (i, j)
                    self.food.grow_food(pos, self.island)
                    if pos in self.herbivores.herbs.keys():
                        self.herbivores.calculate_fitness(pos)
                        self.herbivores.sort_by_fitness(pos)
                        self.herbivores.animals_eat(pos, food_class=self.food)
                        self.herbivores.breeding(pos)
            for i in range(self.island.rader):
                for j in range(self.island.col):
                    pos = (i, j)
                    if pos in self.herbivores.herbs.keys():
                        self.herbivores.aging(pos)
                        self.herbivores.loss_of_weight(pos)
                        self.herbivores.calculate_fitness(pos)
                        self.herbivores.death(pos)

    def add_population(self, population):
        """
        Add a population to the island

        :param population: List of dictionaries specifying population
        """

    @property
    def year(self):
        """Last year simulated."""

    @property
    def num_animals(self):
        """Total number of animals on island."""

    @property
    def num_animals_per_species(self):
        """Number of animals per species in island, as dictionary."""

    @property
    def animal_distribution(self):
        """Pandas DataFrame with animal count per species for each cell on island."""

    def make_movie(self):
        """Create MPEG4 movie from visualization images saved."""

if __name__ == "__main__":
    population = [{'loc' : (3,1),'pop' : [{'species': 'Herbievore','age':2, 'weight': 7.3},
                  {'species': 'Herbievore','age':3, 'weight': 9.3},{'species': 'Herbievore','age':12, 'weight': 7.3},
                                          {'species': 'Herbievore','age':1, 'weight': 7.3},
                                          {'species': 'Herbievore','age':1, 'weight': 7.3},
                                          {'species': 'Herbievore','age':1, 'weight': 7.3},
                                          {'species': 'Herbievore','age':1, 'weight': 7.3},
                                          {'species': 'Herbievore','age':1, 'weight': 7.3},
                                          {'species': 'Herbievore','age':1, 'weight': 7.3},
                                          {'species': 'Herbievore', 'age': 1, 'weight': 7.3}
                                          ]},
                  {'loc' : (2,2),'pop' : [{'species': 'Herbievore','age':3, 'weight': 10},
                  {'species': 'Herbievore','age':4, 'weight': 9},{'species': 'Herbievore','age':5, 'weight': 10}]},
                  {'loc' : (3,3),'pop' : [{'species': 'Herbievore','age':1, 'weight': 10.3}]}]
    seed = 4
    a = BioSim("OOOOO\nOSSSO\nOJJJO\nOJSSO\nOSSSO\nOOOOO", population, seed=seed)
    a.setup_simulation()
    a.simulate(100)
    print(a.herbivores.herbs)
    print(len(a.herbivores.herbs[(3, 1)]))
    print(a.food.food[(3, 1)])
