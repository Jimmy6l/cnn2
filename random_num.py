#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random
import numpy as np
data = np.random.uniform(low=34,high=35,size=(1,15))

for i in data:
    for j in i:
        # print(type(j))
        print(str(round(j,3)),end=",")
        # print(",")

