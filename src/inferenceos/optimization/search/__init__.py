"""Resource-aware, empirically evaluated optimization search.

Import the concrete submodules directly (``inferenceos.optimization.search.runner``,
``...resources``, ``...sampling``, ``...selection``, ``...evaluation``,
``...policy``). This ``__init__`` is intentionally import-light so that
``optimization.policy`` can depend on ``search.policy`` without an import cycle.
"""

from __future__ import annotations
