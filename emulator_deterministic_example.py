import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from numpy.random import random
from functools import lru_cache
import emulator
from models import epi_model_deterministic

t = epi_model_deterministic.get_utility()
print(t)