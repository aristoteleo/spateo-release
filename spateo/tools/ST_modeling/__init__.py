"""
General and generalized linear modeling of spatial transcriptomics

Option to call functions from ST_modeling (e.g. st.tl.ST_modeling.Niche_Model) or directly from Spateo (e.g.
st.tl.Niche_Model).
"""
from .MuSIC import MuSIC
from .MuSIC_downstream import MuSIC_Interpreter
from .MuSIC_upstream import MuSIC_target_selector
from .run_MuSIC import run
from .SWR_mpi import define_spateo_argparse
