# -*- coding: utf-8 -*-

"""
This code will contain the island class
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

import numpy as np

class island:
    def __init__(self, le_map = None):
        self.valid_map_vals = ['O', 'D', 'M', 'S', 'J']
        self.le_map = le_map
        if self.le_map is None:
            self.le_map = "OOO\nOJO\nOOO"
        self.rader = None
        self.col = None
        island.string_to_matrix(self)

    def string_to_matrix(self):
        if type(self.le_map) is not str:
            raise TypeError ('Input needs to be a string')

        list1 = self.le_map.split()
        list2 = []
        for i in range(len(list1)):
            list2 += [a for a in list1[i]]

        self.col = int(len(list2) / len(list1))
        self.rader = int(len(list1))

        self.le_map = np.reshape(list2, (self.col, self.rader))

    def limit_map_vals(self):

        i_max = self.rader - 1
        j_max = self.col - 1

        for i in range(self.rader):
            for j in range(self.col):
                if self.le_map[i, j] not in self.valid_map_vals:
                    raise ValueError('One or more of the terraintypes are not valid')
                elif i == 0 or j == 0 or i == i_max or j == j_max:
                    if self.le_map[i, j] != 'O':
                        raise ValueError('One or more of the perimeter-tiles are not ocean')

    def fetch_map(self):
        return self.le_map

    def fetch_naturetype(self,pos):
        return self.le_map[pos]

class Food(island):
    def __init__(self):



    def øke mat


    def set mat

