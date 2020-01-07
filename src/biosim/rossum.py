# -*- coding: utf-8 -*-

"""
This code will contain the island class
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

import numpy as np

class island:
    def __init__(self, le_map):
        self.valid_map_vals = [O, D, M, S, J]
        self.le_map = le_map

    def string_to_matrix(self):
        list1 = self.map.split()
        list2 = []
        for i in range(len(list1)):
            list2 += [a for a in list1[i]]

        self.col = int(len(list2) / len(list1))
        self.rader = int(len(list1))

        self.le_map = np.reshape(list2, (col, rader))

    def limit_map_vals(self):
        for i in range col:
            for j in range rader:
                if self.le_map[i,j] is not in self.valid_map_vals:
                    raise ValueError('One or more of the terraintypes are not valid')
                elif i == 0 or j == 0 or i = i_max or j = j_max:
                    if self.le_map[i, j] != 'O':
                        raise ValueError('One or more of the perimeter-tiles are not ocean')




    def check_map_values(self):





