# -*- coding: utf-8 -*-

"""
This code will contain the island class
"""

__author__ = "Markus Ola Granheim & Rasmus Svebestad"
__email__ = "mgranhei@nmbu.no & rasmus.svebestad@nmbu.no"

class Cell:
    def __init__(self, fsav_max=None, fjung_max=None, alpha=None, f=None):
        """
        Class for the savannah tiles

        Input:
        :param fsav_max is the maximum amount of food on the savannah tiles
        :param alpha: The growing factor for the food
        :param f: the amount of food the herbivores eat if there is enough food.
        """
        self.food = {}
        self.fsav_max = fsav_max
        self.f = f
        self.alpha = alpha
        self.fjung_max = fjung_max
        if fsav_max is None:
            self.fsav_max = 300

        if f is None:
            self.f = 10

        if alpha is None:
            self.alpha = 0.3

        if fjung_max is None:
            self.fjung_max = 800

    def set_food(self, pos, isle_class):
        if isle_class.fetch_naturetype(pos) == 'S':
            self.food.update({pos: self.fsav_max})
        elif isle_class.fetch_naturetype(pos) == 'J':
            self.food.update({pos: self.fjung_max})
        else:
            self.food.update({pos: 0})

    def grow_food(self, pos, isle_class):
        """
        updates the amount of food after the animals have eaten
        :param pos: The position of the map
        :param isle_class: Takes in the Island class to make use of the fetch_naturetype function
        """
        if isle_class.fetch_naturetype(pos) == 'S':
            self.food.update({pos: self.food[pos] + self.alpha * (self.fsav_max - self.food[pos])})
        elif isle_class.fetch_naturetype(pos) == 'J':
            self.food.update({pos: self.fjung_max})

    def food_gets_eaten(self, pos):
        """
        reduces the amount of food avaliable on the tiles

        :param pos: THe position of the map
        :return: gives out the amount of food eaten
        """
        if self.f <= self.food[pos]:
            self.food[pos] -= self.f
            return self.f
        elif self.food[pos] == 0:
            return 0
        else:
            b = self.food[pos]
            self.food[pos] = 0
            return b
