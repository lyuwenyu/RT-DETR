"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver


from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
}