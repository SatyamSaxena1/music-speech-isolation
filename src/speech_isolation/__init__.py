"""speech_isolation package

This __init__ is intentionally lightweight to avoid importing heavy
dependencies at package import time. Import submodules explicitly when needed,
e.g. `from speech_isolation.train import train`.
"""

__all__ = [
	"train",
	"infer",
	"evaluate",
	"BaseSeparator",
	"si_sdr",
]
