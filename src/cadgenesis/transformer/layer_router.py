"""cadgenesis.transformer.layer_router
====================================
Canonical location for the dynamic layer router used by the self-designing
transformer.

Re-exports :class:`DynamicLayerRouter` from
:mod:`cadgenesis.transformer.self_designing.routing`.
"""

from cadgenesis.transformer.self_designing.routing import DynamicLayerRouter

__all__ = ["DynamicLayerRouter"]
