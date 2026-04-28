import re


# Pattern 1 (most reliable): a structured line like
#   ACTIONS: 0, 19, 23, 66
# Anywhere in the command. We look for this first because it's unambiguous.
# The regex is greedy on the digit-list to capture as much as possible,
# even on the last line of input without a trailing newline.
_STRUCTURED_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?ACTIONS?\s*:\s*([0-9,\s\-]+)",
    re.IGNORECASE,
)

# Pattern 2 (DeepSeek-V4 style): "(action <num>: `ACTION_NAME`)" or
# "(action <num>)" appearing inside numbered list items. Captures the
# integer ID directly, which is more robust than capturing the name.
# Examples that match:
#   "1. **Build a Pylon** (action 19: `BUILD PYLON`)"
#   "2. **Train a Probe** (action 0: `TRAIN PROBE`)"
#   "- **0**: TRAIN PROBE (costs 50 minerals, currently affordable)"
_PARENTHETICAL_ID_PATTERN = re.compile(
    r"\(\s*action\s*(\d+)\s*[:)]",
    re.IGNORECASE,
)

# Pattern 3 (legacy fuzzy): the original pattern from the upstream repo.
# Captures uppercase action names from numbered/bulleted decision lines.
# Used as last-resort fallback for outputs that don't include integer IDs.
_LEGACY_DECISIONS_PATTERN = re.compile(
    r"(?:#+\s*|\*+)?Decisions:?\**[\s\S]*",
    re.IGNORECASE,
)
_LEGACY_ACTIONS_PATTERN = re.compile(
    r"\d+[.:][^\S\n]*[`*]*<?([A-Z][A-Z\s\-]+?)>?[`*]*(?:\s*[-–—*(\n`]|$)"
)

# Pattern 4 (V4-Flash leading-bold): "- **0**: TRAIN PROBE" or
# "**0**: TRAIN PROBE". DeepSeek-V4 favors this format heavily.
_LEADING_ID_BOLD_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?\*\*\s*(\d+)\s*\*\*\s*[:.]?\s*([A-Z][A-Z\s\-]+?)(?:\s*\(|\s*$|\n)"
)

# Pattern 5 (line-leading numeric ID): catches the format DeepSeek-V4 most
# often uses for the final actions list:
#   "0: `TRAIN PROBE` – Continue Probe production"
#   "0: TRAIN PROBE - reasoning..."
#   "1. `0`: **TRAIN PROBE** – ..."
#   "  19: BUILD PYLON –"
# The integer ID is the FIRST run of digits on the line (optionally inside
# backticks, optionally preceded by a list marker like "1." or "-"), followed
# by ":" or "." then the action name. We capture only the ID; the runtime
# resolves it via _resolve_ids, so we don't need to trust the action-name
# spelling for these cases.
_LINE_LEADING_ID_PATTERN = re.compile(
    r"""
    (?:^|\n)             # start of line
    \s*                  # optional indentation
    (?:[-*]\s*)?         # optional bullet
    (?:\d+[\.\)]\s*)?    # optional list-marker like "1."  (non-capturing)
    [`*\s]*              # optional backtick / bold / space
    (\d+)                # CAPTURE: the action id
    [`*]*                # closing backtick / bold
    \s*[:\.]\s*          # separator ":" or "."
    [`*]*                # optional opening of action name decoration
    [A-Z][A-Z\s\-]*[A-Z] # action name (must look like an UPPERCASE name)
    """,
    re.VERBOSE,
)


def _try_structured_extraction(text):
    """Look for an explicit `ACTIONS: 0, 19, 23` line."""
    m = _STRUCTURED_PATTERN.search(text)
    if not m:
        return None
    raw = m.group(1)
    # Stop at the first newline within the captured group — it might have
    # eaten part of the next line if the digit list ran on.
    raw = raw.split('\n', 1)[0]
    out = []
    for piece in raw.split(','):
        piece = piece.strip()
        # Accept only non-negative integers; silently drop "-1", "abc", etc.
        if piece.isdigit():
            out.append(int(piece))
    return out if out else None


def _try_id_based_extraction(text):
    """
    Look for any of three integer-id patterns in the text:
      * "(action <id>...)" parenthetical
      * "- **<id>**: NAME" leading-bold list item
      * "<id>: NAME" / "<id>. NAME" line-leading (most common in DeepSeek-V4)

    Deduplicates while preserving order — the same action_id appearing
    twice in the same command is almost always the LLM repeating itself,
    not actually intending to issue the action twice.

    SPECIAL CASE: if the line-leading IDs form a perfect sequence
    starting at 0 or 1 (0,1,2,3,4 or 1,2,3,4,5), they are almost
    certainly list indices, not action IDs. In that case we discard
    the line-leading matches and fall back to name-based matching.
    """
    seen = set()
    ids = []

    def _add(val):
        try:
            i = int(val)
        except (TypeError, ValueError):
            return
        if i not in seen:
            seen.add(i)
            ids.append(i)

    # Parenthetical and leading-bold are unambiguous — always trust them.
    for m in _PARENTHETICAL_ID_PATTERN.finditer(text):
        _add(m.group(1))
    for m in _LEADING_ID_BOLD_PATTERN.finditer(text):
        _add(m.group(1))

    # Line-leading IDs are ambiguous: could be IDs or list indices.
    line_leading_ids = []
    for m in _LINE_LEADING_ID_PATTERN.finditer(text):
        try:
            line_leading_ids.append(int(m.group(1)))
        except ValueError:
            pass

    if line_leading_ids and _looks_like_list_indices(line_leading_ids):
        # Discard — they're list numbering, not action IDs. Caller will
        # fall through to name-based matching.
        pass
    else:
        for v in line_leading_ids:
            _add(v)

    return ids if ids else None


def _looks_like_list_indices(values):
    """
    Return True if the values look like a markdown list numbering rather
    than a meaningful set of action IDs.

    Heuristic: a strict consecutive sequence starting at 0 or 1, of length
    >= 3, is almost certainly list indices ("0., 1., 2., 3., 4.").
    Real action ID lists from the LLM are not consecutive — they're
    scattered across the action vocabulary (e.g., 0, 19, 22, 23, 66).
    """
    if len(values) < 3:
        return False
    starts_zero = list(range(0, len(values))) == values
    starts_one  = list(range(1, len(values) + 1)) == values
    return starts_zero or starts_one


def _try_legacy_extraction(text):
    """Original upstream regex — name-based fuzzy match."""
    decisions_match = _LEGACY_DECISIONS_PATTERN.search(text)
    if not decisions_match:
        return None
    block = decisions_match.group()
    names = _LEGACY_ACTIONS_PATTERN.findall(block)
    return names if names else None


def extract_actions_from_text(text):
    """
    Backwards-compatible interface — returns a list of action NAMES (strings).
    Used for the legacy code path. Prefer `extract_action_ids_or_names` for
    new callers because it surfaces integer IDs directly when possible.
    """
    legacy = _try_legacy_extraction(text)
    return legacy if legacy is not None else []


def extract_actions_from_command(command, action_extractor, empty_idx, action_db_manager):
    """
    Extract a list of valid action IDs and names from the actor's free-text
    command, using a four-tier strategy:

      Tier 1 — structured: explicit `ACTIONS: 0, 19, 23` line. Most reliable.
      Tier 2 — inline IDs: `(action <num>...)` or `**<num>**: NAME` patterns.
      Tier 3 — legacy fuzzy: original `Decisions:` + uppercase-name regex.
      Tier 4 — vector-DB similarity for any name that didn't directly match.

    Returns (action_ids, action_names) like the original interface.
    Returns ([empty_idx], ["EMPTY ACTION"]) if nothing parseable was found.
    """
    if isinstance(command, list):
        command = " ".join(command)

    # --- Tier 1: structured ACTIONS: ... line ---
    ids = _try_structured_extraction(command)
    if ids:
        valid_ids, valid_names = _resolve_ids(ids, action_extractor)
        if valid_ids:
            return valid_ids, valid_names

    # --- Tier 2: inline integer IDs ---
    ids = _try_id_based_extraction(command)
    if ids:
        valid_ids, valid_names = _resolve_ids(ids, action_extractor)
        if valid_ids:
            return valid_ids, valid_names

    # --- Tier 3: legacy fuzzy name extraction ---
    extracted_decisions = _try_legacy_extraction(command) or []
    if extracted_decisions:
        action_ids, valid_actions = [], []
        for decision in extracted_decisions:
            db_ids, db_actions = action_extractor.extract_and_search_actions(
                decision, action_db_manager
            )
            action_ids.extend(db_ids)
            valid_actions.extend(db_actions)
        if action_ids:
            return action_ids, valid_actions

    # Nothing found anywhere.
    return [empty_idx], ["EMPTY ACTION"]


def _resolve_ids(ids, action_extractor):
    """
    Given a list of integer IDs (possibly noisy / out-of-range), return
    (valid_ids, valid_names) using the extractor's id->name reverse lookup.
    Silently drops IDs that aren't in the action vocabulary.
    """
    # Build id -> name reverse map once (cache on the extractor instance).
    if not hasattr(action_extractor, '_id_to_name_cache'):
        action_extractor._id_to_name_cache = {
            v: k for k, v in action_extractor.full_action_dict.items()
        }
    reverse = action_extractor._id_to_name_cache

    valid_ids, valid_names = [], []
    for aid in ids:
        if aid in reverse:
            valid_ids.append(aid)
            valid_names.append(reverse[aid])
    return valid_ids, valid_names


class ActionExtractor:
    def __init__(self, action_dict):
        self.full_action_dict = {}
        for category in action_dict:
            for key, value in action_dict[category].items():
                self.full_action_dict[value.upper()] = key

    def extract_and_search_actions(self, decision, action_db_manager):
        action = decision.upper()
        if action in self.full_action_dict:
            return [self.full_action_dict[action]], [action]
        else:
            search_results = action_db_manager.search_actions(action)
            if search_results and 'ids' in search_results and 'documents' in search_results:
                actions = search_results['documents']
                if actions:
                    action_ids = search_results['ids']
                    print("vdb_return_action:", actions[0])
                    return [int(action_ids[0])], [actions[0]]
            return [], []