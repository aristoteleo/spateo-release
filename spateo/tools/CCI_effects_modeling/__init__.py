"""
General and generalized linear modeling of spatial transcriptomics

Option to call functions from CCI_effects_modeling (e.g. st.tl.CCI_effects_modeling.Niche_Model) or directly from Spateo (e.g.
st.tl.Niche_Model).
"""

from .MuSIC import MuSIC
from .MuSIC_downstream import MuSIC_Interpreter
from .MuSIC_upstream import MuSIC_Molecule_Selector
from .SWR import define_spateo_argparse
