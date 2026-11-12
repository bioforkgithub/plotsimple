"""Friendly error messages.

simplydata never shows a beginner a confusing library error if it can help it.
Every message says what went wrong, guesses what was meant, and shows what is available.
"""

from __future__ import annotations

import difflib
import functools
import inspect

ISSUES_URL = "https://github.com/bioforkgithub/simplydata/issues"

# Words people may type in either spelling. Left side is accepted, right side is used.
SPELLINGS = {
    "color": "colour",
    "colors": "colours",
    "gray": "grey",
}


class SimplydataError(Exception):
    """A problem explained in plain English."""


class MissingWordError(SimplydataError, AttributeError):
    """Raised when someone asks for a step simplydata does not have.

    It is also an AttributeError so that Python tools such as hasattr() keep working.
    """


def suggest(word: str, options, n: int = 3, cutoff: float = 0.6) -> list[str]:
    """Return the closest matches to `word` among `options`."""
    return difflib.get_close_matches(str(word), [str(o) for o in options], n=n, cutoff=cutoff)


def column_error(name, columns) -> SimplydataError:
    cols = [str(c) for c in columns]
    lines = [f'Column "{name}" not found.']
    guess = suggest(name, cols, n=1)
    if guess:
        lines.append(f'Did you mean "{guess[0]}"?')
    if cols:
        lines.append("Columns you have: " + ", ".join(cols))
    else:
        lines.append("There are no columns yet. Start from a table with data, for example "
                     'sdata.table("your_file.xlsx") or sdata.chart("your_file.xlsx").')
    return SimplydataError("\n".join(lines))


def missing_word_error(word: str, known) -> MissingWordError:
    lines = [f"simplydata has no step called .{word}()."]
    guesses = suggest(word, known)
    if guesses:
        lines.append("Did you mean " + " or ".join(f".{g}()" for g in guesses) + "?")
    lines.append("Type sdata.words() to see every step.")
    lines.append(f"Missing something? Ask for it at {ISSUES_URL}")
    return MissingWordError("\n".join(lines))


def option_error(step: str, option: str, known) -> SimplydataError:
    lines = [f".{step}() has no option called {option}=."]
    guesses = suggest(option, known)
    if guesses:
        lines.append("Did you mean " + " or ".join(f"{g}=" for g in guesses) + "?")
    if known:
        lines.append(f"Options for .{step}(): " + ", ".join(f"{k}=" for k in known))
    return SimplydataError("\n".join(lines))


def choice_error(step: str, option: str, value, choices) -> SimplydataError:
    lines = [f'.{step}() does not know {option}="{value}".']
    guesses = suggest(value, choices, n=1)
    if guesses:
        lines.append(f'Did you mean {option}="{guesses[0]}"?')
    lines.append("You can use: " + ", ".join(f'"{c}"' for c in choices))
    return SimplydataError("\n".join(lines))


# ---------------------------------------------------------------- old names
# When a word or option gets a better name, the old one keeps working for one release.
# The first time it is used in a session, one gentle line says what it is called now.
_NOTED: set[str] = set()


def _quoted(name: str) -> str:
    return name if '"' in name else f'"{name}"'


def renamed_note(old: str, new: str) -> None:
    """Print, once per session, that an old word or option now has a new name."""
    if old in _NOTED:
        return
    _NOTED.add(old)
    print(f"Note: {_quoted(old)} is now called {_quoted(new)}. "
          "The old name will stop working in a later version.")


def old_word(obj, name: str, old_words: dict):
    """The word an old name now stands for (after the note), or None if it is not an old name.

    old_words maps an old name to the new name, or to (new name to show, attribute to use)
    when the old word behaves a little differently from the new one.
    """
    new = old_words.get(name)
    if new is None:
        return None
    shown, attribute = new if isinstance(new, tuple) else (new, new)
    renamed_note(name, shown)
    return getattr(obj, attribute)


def friendly(func=None, *, renamed: dict | None = None):
    """Decorator for every public step.

    * accepts both spellings (color/colour, gray/grey) for option names
    * turns unknown options into a friendly message with a suggestion
    * renamed={"band": "confidence_band"}: an old option name still works, with a note once

    Use it as @friendly, or as @friendly(renamed={...}).
    """
    if func is None:
        return lambda f: friendly(f, renamed=renamed)
    renamed = renamed or {}
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
            if key in renamed and key not in params:
                new = renamed[key]
                renamed_note(f"{key}=", f"{new}=")
                if new in kwargs or new in fixed:
                    raise SimplydataError(f"Give {new}= only once ({key}= is its old name).")
                key = new
            if key not in params and not takes_any:
                raise option_error(func.__name__, key, params)
            fixed[key] = value
        return func(*args, **fixed)

    wrapper._renamed_options = dict(renamed)
    return wrapper
