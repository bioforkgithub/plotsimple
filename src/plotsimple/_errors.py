"""Friendly error messages.

plotsimple never shows a beginner a confusing library error if it can help it.
Every message says what went wrong, guesses what was meant, and shows what is available.
"""

from __future__ import annotations

import difflib
import functools
import inspect

ISSUES_URL = "https://github.com/bioforkgithub/plotsimple/issues"

# Words people may type in either spelling. Left side is accepted, right side is used.
SPELLINGS = {
    "color": "colour",
    "colors": "colours",
    "gray": "grey",
}


class PlotsimpleError(Exception):
    """A problem explained in plain English."""


class MissingWordError(PlotsimpleError, AttributeError):
    """Raised when someone asks for a step plotsimple does not have.

    It is also an AttributeError so that Python tools such as hasattr() keep working.
    """


def suggest(word: str, options, n: int = 3, cutoff: float = 0.6) -> list[str]:
    """Return the closest matches to `word` among `options`."""
    return difflib.get_close_matches(str(word), [str(o) for o in options], n=n, cutoff=cutoff)


def column_error(name, columns) -> PlotsimpleError:
    cols = [str(c) for c in columns]
    lines = [f'Column "{name}" not found.']
    guess = suggest(name, cols, n=1)
    if guess:
        lines.append(f'Did you mean "{guess[0]}"?')
    if cols:
        lines.append("Columns you have: " + ", ".join(cols))
    else:
        lines.append("There are no columns yet. Start from a table with data, for example "
                     'ps.table("your_file.xlsx") or ps.chart("your_file.xlsx").')
    return PlotsimpleError("\n".join(lines))


def missing_word_error(word: str, known) -> MissingWordError:
    lines = [f"plotsimple has no step called .{word}()."]
    guesses = suggest(word, known)
    if guesses:
        lines.append("Did you mean " + " or ".join(f".{g}()" for g in guesses) + "?")
    lines.append("Type ps.words() to see every step.")
    lines.append(f"Missing something? Ask for it at {ISSUES_URL}")
    return MissingWordError("\n".join(lines))


def option_error(step: str, option: str, known) -> PlotsimpleError:
    lines = [f".{step}() has no option called {option}=."]
    guesses = suggest(option, known)
    if guesses:
        lines.append("Did you mean " + " or ".join(f"{g}=" for g in guesses) + "?")
    if known:
        lines.append(f"Options for .{step}(): " + ", ".join(f"{k}=" for k in known))
    return PlotsimpleError("\n".join(lines))


def choice_error(step: str, option: str, value, choices) -> PlotsimpleError:
    lines = [f'.{step}() does not know {option}="{value}".']
    guesses = suggest(value, choices, n=1)
    if guesses:
        lines.append(f'Did you mean {option}="{guesses[0]}"?')
    lines.append("You can use: " + ", ".join(f'"{c}"' for c in choices))
    return PlotsimpleError("\n".join(lines))


def friendly(func):
    """Decorator for every public step.

    * accepts both spellings (color/colour, gray/grey) for option names
    * turns unknown options into a friendly message with a suggestion
    """
    sig = inspect.signature(func)
    params = [p for p in sig.parameters if p != "self"]
    takes_any = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        fixed = {}
        for key, value in kwargs.items():
            for us, uk in SPELLINGS.items():
                if key == us or key.endswith("_" + us):
                    candidate = key[: len(key) - len(us)] + uk
                    if candidate in params:
                        key = candidate
                    break
            if key not in params and not takes_any:
                raise option_error(func.__name__, key, params)
            fixed[key] = value
        return func(*args, **fixed)

    return wrapper
