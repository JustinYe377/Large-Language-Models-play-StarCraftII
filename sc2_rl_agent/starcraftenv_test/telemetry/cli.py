"""Telemetry CLI flags shared by test_the_env.py and multiprocess_test.py."""


def add_telemetry_args(parser):
    """Local JSONL is always written; these flags only control the wandb
    sink and the run config (see README "Telemetry")."""
    parser.add_argument('--wandb_mode', type=str, default=None,
                        choices=['online', 'offline', 'disabled'],
                        help='wandb sink mode. Default: $WANDB_MODE if set, else offline.')
    parser.add_argument('--exp_group', type=str, default='adhoc',
                        help='wandb run group used to compare experiments.')
    parser.add_argument('--LLM_reasoning_effort', type=str, default=None,
                        help='Passed to the API via extra_body when set. Default: not sent.')
    parser.add_argument('--price_in_hit', type=float, default=0.0,
                        help='USD per 1M cache-hit input tokens (0 = no cost estimate).')
    parser.add_argument('--price_in_miss', type=float, default=0.0,
                        help='USD per 1M cache-miss input tokens.')
    parser.add_argument('--price_out', type=float, default=0.0,
                        help='USD per 1M output tokens.')
    return parser
