# -*- coding: utf-8 -*-

"""
This file will test the rossum.py-file
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

from biosim.rossum import island
import pytest
import numpy as np


class TestIsland:
    def test_island_boarder(self):
        """Default constructor callable"""
        a = island()
        assert isinstance(a, island)

    def test_perimeter_error(self):
        """Testing if ValueError is raised when the perimeter contains something other than ocean"""
        a = "OOO\nOJJ\nOOO"
        aa = island(a)
        with pytest.raises(ValueError):
            aa.limit_map_vals()

    def test_unvalid_terrain_error(self):
        """Tests if the map input contains unvalid terrain-types"""
        a = "OOO\nOWO\nOOO"
        aa = island(a)
        with pytest.raises(ValueError):
            aa.limit_map_vals()

    def test_unvaild_type(self):
        """Tests if the code raises TypeError if the map input is not a string"""
        a = [['O', 'O', 'O'], ['O', 'J', 'O'], ['O', 'O', 'O']]
        with pytest.raises(TypeError):
            island(a)

    def test_output_np_array(self):
        """Tests if the output map is right, when input has no error"""
        a = "OOO\nOJO\nOOO"
        aa = island(a)
        el_map = [['O', 'O', 'O'], ['O', 'J', 'O'], ['O', 'O', 'O']]
        assert np.array_equal(aa.fetch_map(), el_map)

    def test_return_neturtype(self):
        a = "OOO\nOJO\nOOO"
        aa = island(a)
        pos = (0,0)
        assert aa.fetch_naturetype(pos) == 'O'
