"""Conservative text rules for explicit policy-sensitive feedback."""

import re
from collections.abc import Iterable

import pandas as pd

_STAFF = r"staff|cashier|guard|security|employee|billing staff|counter staff"
_MISCONDUCT = (
    r"abusive|abuse|misbehav\w*|shout\w*|harass\w*|threaten\w*|insult\w*|"
    r"hit|slap\w*|grab\w*|called me a thief|called .* a thief|chor bola|"
    r"gaali|cheat\w*|deliberately overcharg\w*|kept my change"
)
_MISCONDUCT_PATTERNS = (
    re.compile(
        rf"\b(?:{_STAFF})\b[\s\S]{{0,60}}?\b(?P<signal>{_MISCONDUCT})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<signal>{_MISCONDUCT})\b[\s\S]{{0,60}}?\b(?:{_STAFF})\b",
        re.IGNORECASE,
    ),
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|without|didn['’]?t|doesn['’]?t|don['’]?t|"
    r"did\s+not|does\s+not|isn['’]?t|wasn['’]?t|weren['’]?t)\b"
    r"(?:\s+\w+){0,2}\s*$",
    re.IGNORECASE,
)

_POLICY_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "IT17",
        (
            re.compile(
                r"\b(?:worm\w*|insect\w*|stone|hair|plastic|glass|metal|pin)\b"
                r"[\s\S]{0,45}\b(?:in|inside|within)\b[\s\S]{0,35}"
                r"\b(?:food|product|packet|pack|atta|flour|rice|dal|besan|"
                r"lentils?|cereal|grain|bread|meal)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:food|product|packet|pack|atta|flour|rice|dal|besan|"
                r"lentils?|cereal|grain|bread|meal)\b[\s\S]{0,45}"
                r"\b(?:contains?|had|with)\b[\s\S]{0,25}"
                r"\b(?:worm\w*|insect\w*|stone|hair|plastic|glass|metal|pin)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:dal|atta|rice)\s+madhe\s+(?:kes|baal)\b", re.IGNORECASE),
        ),
    ),
    (
        "IT08",
        (
            re.compile(
                r"\b(?P<signal1>wet|slippery|geela)\s+(?:floor|farsh)\b|"
                r"\b(?:floor|farsh)\b[\s\S]{0,24}\b"
                r"(?P<signal2>wet|slippery|geela|paani)\b|"
                r"\b(?P<signal3>water|paani)\b[\s\S]{0,24}\b(?:floor|farsh)\b|"
                r"\b(?P<signal4>exposed|loose|hanging)\s+wire\b|"
                r"\bwire\b[\s\S]{0,20}\b(?P<signal5>exposed|loose|hanging)\b|"
                r"\b(?P<signal6>blocked|blocked off)\s+(?:fire\s+)?exit\b|"
                r"\b(?P<signal7>falling)\s+(?:stock|goods|items)\b|"
                r"\b(?P<signal8>broken)\s+glass\b|"
                r"\b(?P<signal9>slip(?:ped|ping)?)\s+hazard\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "IT10",
        (
            re.compile(
                r"\b(?:freezer|fridge|chiller|frzr)\b[\s\S]{0,35}"
                r"\b(?P<signal1>not cooling|stopped cooling|warm|thaw\w*|melt\w*|"
                r"temperature|temp|cool aagala|cool nahi)\b|"
                r"\b(?P<signal2>not cooling|stopped cooling|warm|thaw\w*|melt\w*)\b"
                r"[\s\S]{0,35}\b(?:freezer|fridge|chiller|frzr)\b|"
                r"\b(?:ice cream|frozen goods|frozen items|dairy|milk|curd)\b"
                r"[\s\S]{0,30}\b(?P<signal3>melt\w*|thaw\w*|warm|sour)\b|"
                r"\b(?P<signal4>melt\w*|thaw\w*|warm|sour)\b[\s\S]{0,30}"
                r"\b(?:ice cream|frozen goods|frozen items|dairy|milk|curd)\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "IT16",
        (
            re.compile(
                r"\b(?:expired|past\s+(?:the\s+)?expiry|past\s+(?:its\s+)?"
                r"expiration|near[- ]expiry|near[- ]expiration|"
                r"expiry\s+(?:date\s+)?(?:passed|overdue)|"
                r"date\s+(?:has\s+)?(?:passed|expired|nikal\s+gaya))\b",
                re.IGNORECASE,
            ),
        ),
    ),
    (
        "IT06",
        (
            re.compile(
                r"\b(?:cockroach\w*|rat\s+droppings|mouse\s+droppings|"
                r"flies|fly infestation|rodent\w*|keede)\b"
                r"[\s\S]{0,55}\b(?:store|aisle|bakery|food|counter|bin|"
                r"washroom|toilet|shelf|near|inside|around|paas)\b|"
                r"\b(?:store|aisle|bakery|food|counter|bin|washroom|"
                r"toilet|shelf|near|inside|around|paas)\b[\s\S]{0,55}"
                r"\b(?:cockroach\w*|rat\s+droppings|mouse\s+droppings|"
                r"flies|fly infestation|rodent\w*|keede)\b|"
                r"\b(?:dirty|filthy|unclean)\s+(?:food\s+)?"
                r"(?:counter|bakery|washroom|toilet)\b|"
                r"\bfoul\s+smell\b",
                re.IGNORECASE,
            ),
        ),
    ),
)

_ALLERGEN = re.compile(
    r"\b(?:peanut\w*|groundnut\w*|sesame|til|egg|milk|dairy|soy|soya|"
    r"nuts?|almond|cashew|gluten|wheat)\b",
    re.IGNORECASE,
)
_INGREDIENT_CONTEXT = re.compile(r"\b(?:ingredient\w*|allergen\w*)\b", re.IGNORECASE)
_LABEL_CONTEXT = re.compile(
    r"\b(?:label\w*|declared|declare|listed|ingredient\w*|"
    r"veg\s+dot|non[- ]veg|marked\s+(?:veg|vegetarian))\b",
    re.IGNORECASE,
)
_LABEL_ERROR = re.compile(
    r"\b(?:wrong|incorrect|missing|mismatch|doesn['’]?t\s+match|"
    r"does\s+not\s+match|not\s+declared|undeclared|not\s+listed|"
    r"omitted|marked\s+(?:veg|vegetarian))\b",
    re.IGNORECASE,
)


def _has_unnegated_match(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    for pattern in patterns:
        for match in pattern.finditer(text):
            signal_start = next(
                (match.start(name) for name, value in match.groupdict().items() if value),
                match.start(),
            )
            context = text[max(0, signal_start - 30) : signal_start]
            if not _NEGATION.search(context):
                return True
    return False


def _explicit_policy_issue(text: str) -> str | None:
    for issue_type_id, patterns in _POLICY_PATTERNS:
        if _has_unnegated_match(text, patterns):
            return issue_type_id

    has_ingredient_or_allergen = bool(_INGREDIENT_CONTEXT.search(text))
    has_named_allergen = bool(_ALLERGEN.search(text))
    has_label_context = bool(_LABEL_CONTEXT.search(text))
    has_label_error = _has_unnegated_match(text, (_LABEL_ERROR,))
    if has_label_error and (
        has_ingredient_or_allergen or (has_named_allergen and has_label_context)
    ):
        return "IT18"

    if _has_unnegated_match(text, _MISCONDUCT_PATTERNS):
        return "IT04"
    return None


def apply_explicit_policy_rules(texts: pd.Series, issue_type_ids: pd.Series) -> pd.Series:
    """Override model labels only when text explicitly describes a policy-sensitive issue."""
    if len(texts) != len(issue_type_ids):
        raise ValueError("texts and issue_type_ids must have the same length")
    adjusted = [
        _explicit_policy_issue(str(text)) or issue_type_id
        for text, issue_type_id in zip(texts.fillna(""), issue_type_ids)
    ]
    return pd.Series(adjusted, index=issue_type_ids.index, name=issue_type_ids.name)
