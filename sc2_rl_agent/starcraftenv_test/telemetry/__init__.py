"""
Phase-0 experiment telemetry (measurement only, no behaviour changes).

Local JSONL under each game folder is the source of truth; wandb is an
optional sink. Importing this package never imports wandb.
"""

from .schema import SCHEMA_VERSION, LLMCallRecord, DecisionTick, SlotOutcome
from .funnel import fail_cat, FunnelCounter
from .llm_meter import LLMMeter
from .run_logger import RunLogger, resolve_mode
