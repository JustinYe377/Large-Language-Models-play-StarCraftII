"""Protoss bot behaviour version, recorded in every telemetry run config.

v1  original handlers.
v2  handler fixes (success/failure reported correctly, warp-gate and
    disruptor/armor/retreat bugs fixed, probes queue one deep) plus the
    `bot_rules` switch: "dynamic" (default) replaces the hard-coded strategy
    rules, "legacy" keeps them.
"""

BOT_VERSION = "v2"
BOT_RULES = ("dynamic", "legacy")
