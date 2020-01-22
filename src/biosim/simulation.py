# -*- coding: utf-8 -*-

"""
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"


from biosim.island import Island
from biosim.animal import Animal, Herbivores, Carnivores
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
import pandas as pd
import cv2


class BioSim:
    def __init__(self, island_map, ini_pop, seed, ymax_animals=None, cmax_animals=None, img_base=None, img_fmt='png'):
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
        # variables for code/setup/classes
        self.ini_pop = ini_pop
        self.island_map = island_map
        self.tot_num_years = 0

        # variables for movie/save
        if img_base is not None:
            self._img_base = img_base
        else:
            self._img_base = 0

        self._img_fmt = img_fmt
        self._img_ctr = 0

        # Variables for visualization/plot
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.line = None
        self.line2 = None

        if ymax_animals is None:
            self.ymax_animals = 20000
        else:
            self.ymax_animals = ymax_animals
        if cmax_animals is None:
            self.cmax_animals = {'Herbivore': 100, 'Carnivore': 100}
        else:
            self.cmax_animals = cmax_animals

        # Initialization and setting classes
        self.island = Island(island_map)
        self.island.limit_map_vals()
        self.herbivores = Herbivores()
        self.carnivores = Carnivores()
        self.animal = Animal()
        self.setup_simulation()
        np.random.seed(seed)
        random.seed(seed)

    def set_animal_parameters(self, species, params):
        """
        Set parameters for animal species.

        :param species: String, name of animal species
        :param params: Dict with valid parameter specification for species
        """
        if species == 'Herbivore':
            self.herbivores.set_new_params(params)
        else:
            self.carnivores.set_new_params(params)

    def set_landscape_parameters(self, terrain, params):
        """
        Set parameters for landscape types.

        :param terrain: Tells which terrain shall get new parameters
        :param params: Dict with valid parameter specification for landscape
        """
        self.island.set_new_params(terrain, params)

    def setup_simulation(self):
        """
        Function to set up the simulation, sets up the food and inserts the initial population.
        """
        for i in range(self.island.rader):
            for j in range(self.island.col):
                self.island.set_food((i, j))
        self.island.add_animals(self.ini_pop)

    def simulate_one_year(self):
        """
        Function used to simulate one year
        """
        for i in range(self.island.rader):
            for j in range(self.island.col):
                pos = (i, j)
                self.island.grow_food(pos)
                self.herbivores.calculate_fitness(pos, self.island.herbs)
                self.herbivores.sort_by_fitness(pos, self.island.herbs)
                self.herbivores.animals_eat(pos, self.island, self.island.herbs)
                self.herbivores.sort_before_getting_hunted(pos, self.island.herbs)
                self.carnivores.calculate_fitness(pos, self.island.carns)
                self.carnivores.sort_by_fitness(pos, self.island.carns)
                self.carnivores.carnivores_eat(pos, self.island, self.island.carns)
                self.herbivores.breeding(pos, self.island, self.island.herbs)
                self.carnivores.calculate_fitness(pos, self.island.carns)
                self.carnivores.breeding(pos, self.island, self.island.carns)
                self.herbivores.calculate_fitness(pos, self.island.herbs)
                self.carnivores.calculate_fitness(pos, self.island.carns)
        self.herbivores.migration_calculations(self.island, self.island.herbs)
        self.carnivores.migration_calculations(self.island, self.herbivores, self.island.carns)
        self.herbivores.migration_execution(self.island, self.island.herbs)
        self.carnivores.migration_execution(self.island, self.island.carns)
        for i in range(self.island.rader):
            for j in range(self.island.col):
                pos = (i, j)
                self.herbivores.aging(pos, self.island.herbs)
                self.carnivores.aging(pos, self.island.carns)
                self.herbivores.loss_of_weight(pos, self.island.herbs)
                self.carnivores.loss_of_weight(pos, self.island.carns)
                self.herbivores.calculate_fitness(pos, self.island.herbs)
                self.carnivores.calculate_fitness(pos, self.island.carns)
                self.herbivores.death(pos, self.island.herbs)
                self.carnivores.death(pos, self.island.carns)

    def simulate(self, num_years, vis_years=1, img_years=None):
        """
        Function used to simulate num_years, visualizing every vis_years and taking a picture every img_years.

        :param num_years: Number of years to simulate
        :param vis_years: Number that indicates how many years between each visualization
        :param img_years: Number that indicates how many years between taking images, if None a picture will be taken
                          for each visualization
        """
        self._setup_graphics(num_years)
        for year in range(num_years):
            if year % vis_years == 0:
                self._update_graphics()
            if year % img_years == 0:
                self._save_graphics()
            self.simulate_one_year()
            self.tot_num_years += 1

    def add_population(self, population):
        """
        Add a population to the island

        :param population: List of dictionaries specifying population
        """
        self.island.add_animals(population)

    def _setup_graphics(self, num_years):
        """
        This function is used to setup the figure of the grapichs, it creates a figure with 4 subplots and initializes
        the animal lines which are plotted in the upper right subplot.
        :param num_years:
        :return:
        """
        fig = plt.figure()
        axt = fig.add_axes([0.4, 0.8, 0.2, 0.2])
        axt.axis('off')
        self.ax1 = plt.subplot2grid((2, 2), (0, 0))
        self.ax2 = plt.subplot2grid((2, 2), (0, 1))
        self.ax3 = plt.subplot2grid((2, 2), (1, 0))
        self.ax4 = plt.subplot2grid((2, 2), (1, 1))

        self.ax2.set_xlim(self.year + 1, self.year + num_years)
        self.ax2.set_ylim(0, self.ymax_animals)

        self.line = self.ax2.plot(np.arange(num_years + self.year + 1),
                                  np.full(num_years + self.year + 1, np.nan), 'b-')[0]
        self.line2 = self.ax2.plot(np.arange(num_years + self.year + 1),
                                   np.full(num_years + self.year + 1, np.nan), 'r-')[0]

    def _update_graphics(self):
        """
        :param year: Which year are we printing graphics for
        Updates graphics with current data.
        """
        plt.suptitle('Years since the simulation started: ' + str(self.year) + ' years')
        rgb_value = {"O": mcolors.to_rgba("navy"),
                     "J": mcolors.to_rgba("forestgreen"),
                     "S": mcolors.to_rgba("#e1ab62"),
                     "D": mcolors.to_rgba("yellow"),
                     "M": mcolors.to_rgba("lightslategrey")
                     }

        kart_rgb = [[rgb_value[column] for column in row]
                    for row in self.island_map.split()]

        self.ax1.imshow(kart_rgb)
        self.ax1.set_xticks(range(len(kart_rgb[0])))
        self.ax1.set_xticklabels(range(1, 1 + len(kart_rgb[0])))
        self.ax1.set_yticks(range(len(kart_rgb)))
        self.ax1.set_yticklabels(range(1, 1 + len(kart_rgb)))
        self.ax1.axis('scaled')
        self.ax1.set_title('Map')

        # Upper right subplot
        ydata = self.line.get_ydata()
        ydata2 = self.line2.get_ydata()
        ydata[self.year] = self.num_animals_per_species['Herbivore']
        ydata2[self.year] = self.num_animals_per_species['Carnivore']
        self.line.set_ydata(ydata)
        self.line2.set_ydata(ydata2)
        self.ax2.set_title('Total number of animals')

        self.ax3.imshow(self.animal_distribution_for_plot[:, :, 0], 'Greens', vmax=self.cmax_animals['Herbivore'])
        self.ax3.set_xticks(range(len(kart_rgb[0])))
        self.ax3.set_xticklabels(range(1, 1 + len(kart_rgb[0])))
        self.ax3.set_yticks(range(len(kart_rgb)))
        self.ax3.set_yticklabels(range(1, 1 + len(kart_rgb)))
        self.ax3.axis('scaled')
        self.ax3.set_title('Herbivore distribution: ' + str(self.num_animals_per_species['Herbivore']))

        self.ax4.imshow(self.animal_distribution_for_plot[:, :, 1], 'Reds', vmax=self.cmax_animals['Carnivore'])
        self.ax4.set_xticks(range(len(kart_rgb[0])))
        self.ax4.set_xticklabels(range(1, 1 + len(kart_rgb[0])))
        self.ax4.set_yticks(range(len(kart_rgb)))
        self.ax4.set_yticklabels(range(1, 1 + len(kart_rgb)))
        self.ax4.axis('scaled')
        self.ax4.set_title('Carnivore distribution: ' + str(self.num_animals_per_species['Carnivore']))
        plt.pause(1e-9)

    def _save_graphics(self):
        """Saves graphics to file if file name given."""

        if self._img_base is None:
            return

        plt.savefig('{base}_{num:05d}.{type}'.format(base=self._img_base,
                                                     num=self._img_ctr,
                                                     type=self._img_fmt))
        self._img_ctr += 1

    def make_movie(self):
        import glob

        img_array = []
        for filename in glob.glob(self._img_base + '*' + self._img_fmt):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 6, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    @property
    def year(self):
        """Last year simulated."""
        return self.tot_num_years

    @property
    def num_animals(self):
        """Total number of animals on island."""
        return sum(self.island.num_animals())

    @property
    def num_animals_per_species(self):
        """Number of animals per species in island, as dictionary."""
        herb, carn = self.island.num_animals()
        return {'Herbivore': herb, 'Carnivore': carn}

    @property
    def animal_distribution_for_plot(self):
        """Pandas DataFrame with animal count per species for each cell on island."""
        animal_list = np.zeros((self.island.rader, self.island.col, 2))
        for i in range(self.island.rader):
            for j in range(self.island.col):
                if (i, j) in self.island.herbs.keys():
                    animal_list[i, j, 0] = len(self.island.herbs[(i, j)])
                if (i, j) in self.island.carns.keys():
                    animal_list[i, j, 1] = len(self.island.carns[(i, j)])
        return animal_list

    @property
    def animal_distribution(self):
        a = np.zeros((self.island.rader * self.island.col, 4))
        for i in range(self.island.rader):
            a[i * self.island.col:(i + 1) * self.island.col, 0] = i

        c = np.arange(self.island.col)
        for i in range(self.island.rader):
            a[i * self.island.col:(i + 1) * self.island.col, 1] = c

        for pos in self.island.herbs.keys():
            a[pos[0] * self.island.col + pos[1], 2] += len(self.island.herbs[pos])

        for pos in self.island.carns.keys():
            a[pos[0] * self.island.col + pos[1], 3] += len(self.island.carns[pos])

        return pd.DataFrame(a, columns=['Row', 'Col', 'Herbivore', 'Carnivore'])


if __name__ == "__main__":
    # -*- coding: utf-8 -*-

    import textwrap

    from biosim.simulation import BioSim

    plt.ion()

    geogr = """\
               OOOOOOOOOOOOOOOOOOOOO
               OOOOOOOOSMMMMJJJJJJJO
               OSSSSSJJJJMMJJJJJJJOO
               OSSSSSSSSSMMJJJJJJOOO
               OSSSSSJJJJJJJJJJJJOOO
               OSSSSSJJJDDJJJSJJJOOO
               OSSJJJJJDDDJJJSSSSOOO
               OOSSSSJJJDDJJJSOOOOOO
               OSSSJJJJJDDJJJJJJJOOO
               OSSSSJJJJDDJJJJOOOOOO
               OOSSSSJJJJJJJJOOOOOOO
               OOOSSSSJJJJJJJOOOOOOO
               OOOOOOOOOOOOOOOOOOOOO"""
    geogr = textwrap.dedent(geogr)

    ini_herbs = [
        {
            "loc": (10, 10),
            "pop": [
                {"species": "Herbivore", "age": 5, "weight": 20}
                for _ in range(150)
            ],
        }
    ]
    ini_carns = [
        {
            "loc": (10, 10),
            "pop": [
                {"species": "Carnivore", "age": 5, "weight": 20}
                for _ in range(40)
            ],
        }
    ]

    sim = BioSim(island_map=geogr, ini_pop=ini_herbs, seed=123456, img_base='movie_images/')

    sim.set_animal_parameters("Herbivore", {"zeta": 3.2, "xi": 1.8})
    sim.set_animal_parameters(
        "Carnivore",
        {
            "a_half": 70,
            "phi_age": 0.5,
            "omega": 0.3,
            "F": 65,
            "DeltaPhiMax": 9.0,
        },
    )
    sim.set_landscape_parameters("J", {"f_max": 700})

    sim.simulate(num_years=100, vis_years=1, img_years=1)

    sim.add_population(population=ini_carns)
    sim.simulate(num_years=100, vis_years=1, img_years=1)
    sim.make_movie()
    input("Press ENTER")
