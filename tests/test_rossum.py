# -*- coding: utf-8 -*-

"""
This file will test the rossum.py-file
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

def test_empty_island():
    """Empty island can be created"""
    BioSim(island_map="OO\nOO", ini_pop=[], seed=1)


def test_minimal_island():
    """Island of single jungle cell"""
    BioSim(island_map="OOO\nOJO\nOOO", ini_pop=[], seed=1)


def test_all_types():
    """All types of landscape can be created"""
    BioSim(island_map="OOOO\nOJSO\nOMDO\nOOOO", ini_pop=[], seed=1)

