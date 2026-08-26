try:
    from .py_blockSQP import *
except ImportError as IERR:
    if IERR.msg[:len("generic_type: type ")] == "generic_type: type ":
        raise ImportError(IERR.msg + "\n**Note**: The above error likely ocurred because a different version of py_blockSQP was previously loaded. This is due to how Python handles pybind11/boost::python modules. Start a new Python session.") from None
    else:
        raise IERR
# from .py_blockSQP import *
from .blockSQP_Problemspec import BlockSQP_Problem as Problemspec