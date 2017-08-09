# -*- coding: utf-8 -*-
# Maximilian Christ (max.christ@me.com)

import Unittest
from kNN_DTW import _finite_of, EOTS
import numpy as np


class Test_kNN_DTW(Unittest):

    def test__finite_of(self):
        x = np.random.normal(size=100)
        self.assertEqual(_finite_of(x), x)
        self.assertEqual(_finite_of([0, 0, EOTS]), np.array([0, 0]))
        self.assertEqual(_finite_of([EOTS, EOTS]), np.array([]))
