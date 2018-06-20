import networkx as nx
import matplotlib.pyplot as plt
import json

import numpy as np
from keras.layers import Activation, Dense, Input, initializers 
from keras.models import model_from_json,  load_model
from keras import optimizers