"""
Spatial lag model for the purpose of predicting and quantifying ligand:receptor interactions.

Builds from the implementation in spreg: https://spreg.readthedocs.io/en/latest/,
Authors: Luc Anselin, David C. Folch
"""
from spreg.twosls_sp import BaseGM_Lag


class LR_GM_lag(BaseGM_Lag):



# Need to modify the spreg implementation b/c the base appends the spatial lag of y onto the spatial lag of anything
# else and in the case of L:R prediction that might not always be desirable...
