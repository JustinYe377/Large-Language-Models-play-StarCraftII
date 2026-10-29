"""CLI flags for HarnessAgent (--agent_type harness / deepseek_harness)."""


def add_harness_args(parser):
    parser.add_argument('--LLM_thinking', type=str, default='disabled', choices=['disabled', 'enabled', 'auto'],
                        help='Hidden reasoning for harness agents. "auto" sends nothing (server default).')
    parser.add_argument('--LLM_thinking_provider', type=str, default='auto',
                        choices=['auto', 'deepseek', 'chat_template', 'none'],
                        help='How to send the thinking switch: DeepSeek "thinking" field, or vLLM/SGLang '
                             'chat_template_kwargs.enable_thinking. auto: by --LLM_api_base.')
    parser.add_argument('--structured_mode', type=str, default='auto',
                        choices=['auto', 'tools', 'json_schema', 'json_object', 'text'],
                        help='How decisions are constrained. auto: strongest the server accepts.')
    parser.add_argument('--harness_max_actions', type=int, default=5,
                        help='Maximum actions per decision (K).')
    return parser
