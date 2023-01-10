"""
General and generalized linear modeling of spatial transcriptomics

Option to call functions from ST_regression (e.g. st.tl.ST_regression.Niche_Model) or directly from Spateo (e.g.
st.tl.Niche_Model).
"""
from .spatial_regression import (
    Category_Model,
    Lagged_Model,
    Niche_LR_Model,
    Niche_Model,
)
