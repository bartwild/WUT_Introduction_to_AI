import random

random.seed(42137)

ATTRS_NAMES = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
CLASS_VALUES = ["not_recom", "recommend", "very_recom", "priority", "spec_prior"]
N_ATTRS = 6
PERCENT_OF_DRAWN_ROWS = 100
N_TREES = 40
MAX_DEPTH = 0
PERCENT_OF_TRAIN_DATA = 8
ATTR_TO_INDEX = {ATTRS_NAMES[i]: i for i in range(len(ATTRS_NAMES))}
SHUFFLED_TRAIN_DATA = False
