# Compatibility shim: re-export symbols from `Intermediate` package
# This allows imports using lowercase `intermediate` to work while
# the real implementation resides in `Intermediate/`.

try:
	# Import everything from the real package
	from Intermediate import *
	from Intermediate import __all__ as _real_all
except Exception:
	# If Intermediate is not available, provide minimal fallbacks
	_real_all = []

# Export names
__all__ = list(_real_all)

# Also re-export modules lazily for attribute access
import importlib
import sys as _sys

class _ModuleProxy:
	def __getattr__(self, name):
		try:
			mod = importlib.import_module('Intermediate')
			return getattr(mod, name)
		except Exception:
			raise AttributeError(name)

# Provide module-level attribute access
_proxy = _ModuleProxy()

def __getattr__(name):
	return getattr(_proxy, name)
