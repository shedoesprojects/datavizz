"""Registry for plot functions and a simple generator entrypoint."""

from typing import Callable, Dict, Any, List, Tuple
from types import FunctionType

_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_plot(name: str, params: List[str] = None, engines: List[str] = None):
    params = params or []
    engines = engines or ["matplotlib"]
    def decorator(func: Callable):
        _REGISTRY[name] = {"func": func, "params": params, "engines": engines}
        return func
    return decorator

def get_plot(name: str) -> Dict[str, Any]:
    return _REGISTRY.get(name)

def list_plots() -> List[str]:
    return sorted(_REGISTRY.keys())

def plot_metadata(name: str) -> Dict[str, Any]:
    return _REGISTRY.get(name, {})

def generate_plot(df, plot_name: str, params: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Execute the registered plot. Returns (engine, payload) where:
      - engine == "plotly" -> payload is a plotly figure
      - engine == "matplotlib" -> payload is PNG bytes
    """
    meta = get_plot(plot_name)
    if meta is None:
        raise ValueError(f"Plot '{plot_name}' is not registered.")
    func = meta["func"]
    # take engine from params or default to first allowed
    engine = params.pop("engine", meta["engines"][0])
    return func(df, engine=engine, **params)
