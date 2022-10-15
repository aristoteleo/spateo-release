"""
General and generalized linear modeling of spatial transcriptomics

Option to call functions from ST_regression (e.g. st.tl.ST_regression.Niche_Interpreter) or directly from Spateo (e.g.
st.tl.Niche_Interpreter).
"""
from .spatial_regression import (
    Category_Interpreter,
    Ligand_Lagged_Interpreter,
    Niche_Interpreter,
    Niche_LR_Interpreter,
)