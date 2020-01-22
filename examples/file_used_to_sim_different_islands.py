# -*- coding: utf-8 -*-

import textwrap
from matplotlib import pyplot as plt
from biosim.simulation import BioSim

"""
Compatibility check for BioSim simulations.

This script is based heavily on Hans Ekkehard Plessers check_sim.py-file in this folder. This script will be used to 
simulate different islands with different parameters.
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasv@nmbu.no"


plt.ion()

geogr = """\
           OOOOOOOOOOOOOOOOOOOOO
           OOOOJJJJJMMMMJJJJJJJO
           OSSSSSJJJJJJJJJJJJJOO
           OSSSSSSSSSMMJJJJJJOOO
           OSSSSSJJJJJJJJJJJJOOO
           OSSSSSJJJDDJJJSJJJOOO
           OSSJJJJJDDDJJJSSSSOOO
           OOSSSSJJJDDJJJSOOOOOO
           OSSSJJJJJDDJJJJJJJOOO
           OJJJJJJJJDDJJJJOOOOOO
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

sim = BioSim(island_map=geogr, ini_pop=ini_herbs, seed=123456, img_base='checksim_img/')

sim.set_animal_parameters("Herbivore", {"zeta": 3.2, "xi": 1.8})
sim.set_animal_parameters(
    "Carnivore",
    {
        "a_half": 50,
        "phi_age": 0.5,
        "omega": 0.3,
        "F": 55,
        "DeltaPhiMax": 9.0,
    },
)
sim.set_landscape_parameters("J", {"f_max": 700})

sim.simulate(num_years=30, vis_years=1, img_years=1)

sim.add_population(population=ini_carns)
sim.simulate(num_years=30, vis_years=1, img_years=1)
sim.make_movie()
input("Press ENTER")
