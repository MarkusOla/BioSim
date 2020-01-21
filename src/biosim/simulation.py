# -*- coding: utf-8 -*-

"""
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"


from biosim.rossum import Island
from biosim.rossum_to import Animal, Herbivores, Carnivores
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import subprocess
import pandas as pd
import os



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
        self.ymax_animals = ymax_animals
        self.cmax_animals = cmax_animals
        self.img_fmt = img_fmt
        self.island_map = island_map
        self.island = Island(island_map)
        self.island.limit_map_vals()
        self.herbivores = Herbivores()
        self.carnivores = Carnivores()
        self.animal = Animal()
        self.setup_simulation()

        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self._fig = None
        self._map_ax = None
        self._animal_ax = None
        self._herbivore_line = None
        self._carnivore_line = None
        self._herbivore_ax = None
        self._herbivore_axis = None
        self._carnivore_ax = None
        self._carnivore_axis = None
        self._max_animals = None
        self._island_info_ax = None
        self._island_info_txt = None
        self._img_fmt = img_fmt
        self._step = 0
        self._final_step = None
        self._img_ctr = 0
        self.line = None
        self.line2 = None

        # the following will be initialized by _setup_graphics
        self._fig = None
        self._map_ax = None
        self._img_axis = None
        self._mean_ax = None
        self._mean_line = None
        np.random.seed(seed)

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
        Set parameters for landscape type.

        :param params: Dict with valid parameter specification for landscape
        """
        self.island.set_new_params(terrain, params)

    def setup_simulation(self):
        for i in range(self.island.rader):
            for j in range(self.island.col):
                self.island.set_food((i, j))
        self.island.add_animals(self.ini_pop)

    def simulate_one_year(self):
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
                self.carnivores.breeding(pos, self.island, self.island.carns)
                self.herbivores.calculate_fitness(pos, self.island.herbs)
                self.carnivores.calculate_fitness(pos, self.island.carns)
        self.herbivores.migration_calculations(self.island.rader, self.island.col,
                                               self.island, self.island.herbs)
        self.carnivores.migration_calculations(self.island.rader, self.island.col,
                                               self.island, self.herbivores, self.island.carns)
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

        animals = []
        self._setup_graphics(num_years)
        for year in range(num_years):
            self.simulate_one_year()
            animals.append(self.num_animals)
            self._update_graphics(num_years, year)

    def add_population(self, population):
        """
        Add a population to the island

        :param population: List of dictionaries specifying population
        """
        self.island.add_animals(population)

    def make_movie(self, movie_fmt='mp4'):
        """
        Creates MPEG4 movie from visualization images saved.
        .. :note:
            Requires ffmpeg
        The movie is stored as img_base + movie_fmt
        """

        if self._img_base is None:
            raise RuntimeError("No filename defined.")

        if movie_fmt == 'mp4':
            try:
                # Parameters chosen according to http://trac.ffmpeg.org/wiki/Encode/H.264,
                # section "Compatibility"
                subprocess.check_call([_FFMPEG_BINARY,
                                       '-i', '{}_%05d.png'.format(self._img_base),
                                       '-y',
                                       '-profile:v', 'baseline',
                                       '-level', '3.0',
                                       '-pix_fmt', 'yuv420p',
                                       '{}.{}'.format(self._img_base,
                                                      movie_fmt)])
            except subprocess.CalledProcessError as err:
                raise RuntimeError('ERROR: ffmpeg failed with: {}'.format(err))
        elif movie_fmt == 'gif':
            try:
                subprocess.check_call([_CONVERT_BINARY,
                                       '-delay', '1',
                                       '-loop', '0',
                                       '{}_*.png'.format(self._img_base),
                                       '{}.{}'.format(self._img_base,
                                                      movie_fmt)])
            except subprocess.CalledProcessError as err:
                raise RuntimeError('ERROR: convert failed with: {}'.format(err))
        else:
            raise ValueError('Unknown movie format: ' + movie_fmt)

    def _update_system_map(self, sys_map):
        '''Update the 2D-view of the system.'''

        if self._img_axis is not None:
            self._img_axis.set_data(sys_map)
        else:
            self._img_axis = self._map_ax.imshow(sys_map,
                                                 interpolation='nearest',
                                                 vmin=0, vmax=1)
            plt.colorbar(self._img_axis, ax=self._map_ax,
                         orientation='horizontal')

    def _update_mean_graph(self, mean):
        ydata = self._mean_line.get_ydata()
        ydata[self._step] = mean
        self._mean_line.set_ydata(ydata)

    def _setup_graphics(self, num_years):
        fig = plt.figure()
        axt = fig.add_axes([0.4, 0.8, 0.2, 0.2])
        axt.axis('off')
        self.ax1 = plt.subplot2grid((2, 2), (0, 0))
        self.ax2 = plt.subplot2grid((2, 2), (0, 1))
        self.ax3 = plt.subplot2grid((2, 2), (1, 0))
        self.ax4 = plt.subplot2grid((2, 2), (1, 1))

        self.ax2.set_xlim(0, num_years)
        self.ax2.set_ylim(0, 10000)

        self.line = self.ax2.plot(np.arange(num_years),
                                  np.full(num_years, np.nan), 'b-')[0]
        self.line2 = self.ax2.plot(np.arange(num_years),
                                   np.full(num_years, np.nan), 'r-')[0]

    def _update_graphics(self, num_years, year):
        """Updates graphics with current data."""
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
        herbivores, carnivores = self.island.num_animals()
        ydata[year] = herbivores
        ydata2[year] = carnivores
        self.line.set_ydata(ydata)
        self.line2.set_ydata(ydata2)
        self.ax2.set_title('Total number of animals')

        self.ax3.imshow(self.animal_distribution[:, :, 0], 'Greens')
        self.ax3.set_xticks(range(len(kart_rgb[0])))
        self.ax3.set_xticklabels(range(1, 1 + len(kart_rgb[0])))
        self.ax3.set_yticks(range(len(kart_rgb)))
        self.ax3.set_yticklabels(range(1, 1 + len(kart_rgb)))
        self.ax3.axis('scaled')
        self.ax3.set_title('Herbivore distribution: ' + str(self.num_animals_per_species[0]))

        self.ax4.imshow(self.animal_distribution[:, :, 1], 'Reds')
        self.ax4.set_xticks(range(len(kart_rgb[0])))
        self.ax4.set_xticklabels(range(1, 1 + len(kart_rgb[0])))
        self.ax4.set_yticks(range(len(kart_rgb)))
        self.ax4.set_yticklabels(range(1, 1 + len(kart_rgb)))
        self.ax4.axis('scaled')
        self.ax4.set_title('Carnivore distribution: ' + str(self.num_animals_per_species[1]))
        self._save_graphics()
        plt.pause(1e-4)

    def _save_graphics(self):
        """Saves graphics to file if file name given."""
        base = 'checksim_img/'

        plt.savefig('{base}_{num:05d}.{type}'.format(base=base, num=self._img_ctr, type=self._img_fmt))
        self._img_ctr += 1

    @property
    def year(self):
        """Last year simulated."""

    @property
    def num_animals(self):
        """Total number of animals on island."""
        return sum(self.island.num_animals())

    @property
    def num_animals_per_species(self):
        """Number of animals per species in island, as dictionary."""
        return self.island.num_animals()

    @property
    def animal_distribution(self):
        """Pandas DataFrame with animal count per species for each cell on island."""
        animal_list = np.zeros((self.island.rader, self.island.col, 2))
        for i in range(self.island.rader):
            for j in range(self.island.col):
                if (i, j) in self.island.herbs.keys():
                    animal_list[i, j, 0] = len(self.island.herbs[(i, j)])
                if (i, j) in self.island.carns.keys():
                    animal_list[i, j, 1] = len(self.island.carns[(i, j)])
        return animal_list


    def make_movie(self):
        """Create MPEG4 movie from visualization images saved."""
        image_folder = 'BioSim_G25_Granheim_Svebestad\examples\checksim_img'
        video_name = 'video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":

    a.make_movie