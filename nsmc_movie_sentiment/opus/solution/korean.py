"""Korean text utilities: Hangul jamo decomposition (pure python, no external data)."""

CHO = "\u3131\u3132\u3134\u3137\u3138\u3139\u3141\u3142\u3143\u3145\u3146\u3147\u3148\u3149\u314a\u314b\u314c\u314d\u314e"
JUNG = (
    "\u314f\u3150\u3151\u3152\u3153\u3154\u3155\u3156\u3157\u3158\u3159\u315a"
    "\u315b\u315c\u315d\u315e\u315f\u3160\u3161\u3162\u3163"
)
JONG = (
    "\u3131\u3132\u3133\u3134\u3135\u3136\u3137\u3139\u313a\u313b\u313c\u313d\u313e\u313f"
    "\u3140\u3141\u3142\u3144\u3145\u3146\u3147\u3148\u314a\u314b\u314c\u314d\u314e"
)
FILL = "\u007c"  # '|' placeholder for empty jongseong -> fixed 3 chars / syllable

_BASE = 0xAC00
_LAST = 0xD7A3

# Precompute decomposition table for all 11172 Hangul syllables.
_TABLE = {}
for _code in range(_BASE, _LAST + 1):
    _i = _code - _BASE
    _TABLE[chr(_code)] = CHO[_i // 588] + JUNG[(_i % 588) // 28] + (
        JONG[(_i % 28) - 1] if (_i % 28) else FILL
    )


def to_jamo(text):
    """Decompose Hangul syllables into fixed-width 3-jamo groups; other chars pass through."""
    out = []
    get = _TABLE.get
    for ch in text:
        out.append(get(ch, ch))
    return "".join(out)


def to_jamo_list(texts):
    return [to_jamo(t) for t in texts]
