"""core.contracts — All system contracts."""
from .execution import IExecution
from .governance import IGovernance
from .intelligence import IIntelligence
from .logger import ILogger
from .observability import IObservability
from .persistence import IPersistence
from .state import IState
from .time import ITime
from .translation import ITranslation

__all__ = ["IState","ILogger","ITime","IGovernance","IExecution",
           "IObservability","IPersistence","IIntelligence","ITranslation"]
