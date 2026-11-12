"""Write docs/cheatsheet.md from the same word list that ps.words() prints.

Run it after adding or changing a word:

    python docs/make_cheatsheet.py

A test (tests/test_learn.py) fails if cheatsheet.md is out of date.
"""

from __future__ import annotations

from pathlib import Path

from plotsimple import learn

HERE = Path(__file__).resolve().parent
OUT = HERE / "cheatsheet.md"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def make() -> str:
    aliases = learn._aliases()
    lines = [
        "# plotsimple cheat sheet",
        "",
        "Every word on one page. Most words are steps on a chart:",
        "",
        "```python",
        "import plotsimple as ps",
        'marks = ps.example("marks")        # a made-up table to practise with',
        "chart = ps.chart(marks)            # or ps.chart(\"my_file.xlsx\")",
        'chart.bars(x="name", y="marks")',
        'chart.title_top("My class marks")',
        'chart.save("marks.png")',
        "```",
        "",
        'In Python, `ps.help("bars")` explains any word, `ps.words()` prints this list and',
        "`ps.lesson(1)` starts the one-day course.",
        "",
        "Every example runs as written on the practice table in the *Table* column:",
        'for example `weather = ps.example("weather")` then `chart = ps.chart(weather)`.',
        "Steps such as `fit_line` or `title_top` go after a chart step such as `chart.dots(...)`,",
        "and `a`, `b` in `ps.side_by_side(a, b)` are charts you made first.",
        "",
        "*This file is made by `docs/make_cheatsheet.py`. Do not edit it by hand.*",
    ]
    for group, keys in learn.GROUPS:
        lines += ["", f"## {group}", "", "| Word | Example | Table | What it does |",
                  "|---|---|---|---|"]
        for key in keys:
            word = f"`{key}`"
            if key in aliases:
                word += " (also " + ", ".join(f"`{a}`" for a in aliases[key]) + ")"
            example = f"`{_cell(learn._short_example(key))}`"
            table = learn._example_table(key) or ""
            lines.append(f"| {word} | {example} | {table} | {_cell(learn._summary(key))} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.write_text(make(), encoding="utf-8")
    print(f"Wrote {OUT}")
