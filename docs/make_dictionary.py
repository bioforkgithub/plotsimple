"""Build the simplydata dictionary (docs/manual/simplydataDictionary.tex and .pdf) from the code.

The dictionary lists every simplydata word A to Z, with what it means, a short example, its main
options, the names other tools use for it, related words, and the page of the manual where it is
explained (and the lesson that teaches it). A second part, "Coming from another tool?", lists every
name from other tools (iloc, groupby, curve_fit, hist ...) and every old name, each pointing to the
simplydata word.

Everything comes from simplydata itself, as for the manual and the cheat sheet: the words, their
groups, summaries and examples from simplydata/learn.py, the options from each word's signature,
the other names from simplydata/_other_names.py and the old names from the words themselves.
Only the short texts on the first pages and the word families below are written by hand.

The page numbers are read from the manual's .aux file, so the manual must be compiled first.
Building the manual builds the dictionary too:

    python docs/make_manual.py --no-run    # compile the manual, then write and compile the dictionary
    python docs/make_dictionary.py         # only the dictionary, from the last compiled manual

A test (tests/test_learn.py) fails if a word has no page, or if the dictionary was made from an
older manual than docs/manual/simplydata_manual.tex.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_manual  # noqa: E402  (docs/make_manual.py: the same LaTeX helpers and word labels)
from make_manual import code, esc, rich  # noqa: E402

import simplydata  # noqa: E402
from simplydata import learn  # noqa: E402
from simplydata._other_names import OTHER_NAMES  # noqa: E402

MANUAL = make_manual.MANUAL
TEX = MANUAL / "simplydataDictionary.tex"
PDF = MANUAL / "simplydataDictionary.pdf"
MANUAL_PDF_NAME = make_manual.PDF.name  # links open the manual next to the dictionary

# How each kind of word is written in front of the word, and the order of the senses of a headword.
WRITTEN = {"sdata": "sdata", "table": "table", "chart": "chart", "option": "chart", "numbers": "nums",
           "grid": "grid", "dist": "dist"}
KIND_ORDER = ["sdata", "table", "chart", "option", "numbers", "grid", "dist"]
KIND_NAME = {"sdata": "simplydata", "table": "table", "chart": "chart", "option": "option in a chart step",
             "numbers": "numbers", "grid": "grid of numbers", "dist": "distribution"}
MAX_OPTIONS = 8  # the main options: the first ones in the brackets, then "and N more"
MAX_SEE_ALSO = 5

# ---------------------------------------------------------------- word families (hand-written)
# (how the family is written, what it means in plain words, which words belong to it). The words are
# found in the code, so a new keep_ word joins its family by itself.
FAMILIES = [
    ("keep_...", "keeps only some rows, columns or numbers; the rest go",
     lambda w: w.startswith("keep_")),
    ("remove_...", "takes some rows, columns or numbers away; the rest stay",
     lambda w: w.startswith("remove_")),
    ("add_..._column", "adds a new column, made from the columns you have",
     lambda w: w.startswith("add_") and w.endswith("_column")),
    ("add_...", "adds something else to a table: rows, groups, new axes",
     lambda w: w.startswith("add_") and not w.endswith("_column")),
    ("make_...", "changes what a column holds: numbers, text, dates, small or capital letters",
     lambda w: w.startswith("make_")),
    ("show_... and hide_...", "looks at the data, or a part of the chart, without changing anything",
     lambda w: w.startswith(("show_", "hide_"))),
    ("count_...", "counts rows, values, empty cells or numbers",
     lambda w: w.startswith("count_")),
    ("..._of", "sums up one column in one number (or one number for each group)",
     lambda w: w.endswith("_of")),
    ("compare_...", "a statistical test: is the difference bigger than chance? It names the test and gives "
     "the p-value", lambda w: w.startswith("compare_")),
    ("fit_...", "finds the line, curve, formula or distribution that best matches the data",
     lambda w: w.startswith("fit_")),
    ("find_... and solve", "finds where a formula is smallest or biggest, or where it equals something",
     lambda w: w.startswith("find_") or w == "solve"),
    ("..._x and ..._y", "a chart word for one axis: x goes across, y goes up",
     lambda w: w.endswith(("_x", "_y"))),
    ("legend_...", "where the legend goes", lambda w: w.startswith("legend_")),
    ("line_... and shade_...", "draws a line or a shaded band across the chart",
     lambda w: w.startswith(("line_at", "line_with", "shade_"))),
    ("as_...", "hands the data on to another library", lambda w: w.startswith("as_")),
    ("undo, steps, start_again", "every change can be taken back",
     lambda w: w in ("undo", "steps", "start_again")),
]

FRONT = r"""
\begin{titleblock}
{\fontsize{30}{34}\selectfont\sffamily\bfseries\color{brand} simplydata dictionary\par}
\vspace{0.4em}
{\large\sffamily Every word, A to Z, with the page of the manual that explains it\par}
\vspace{0.3em}
{\sffamily\small\color{inkgrey} Version @VERSION@ \textbullet{} @WORDS@ words \textbullet{} @OTHERS@ names from other tools
\textbullet{} made from the code by \code{docs/make\_dictionary.py}\par}
\end{titleblock}

\section*{How to use this dictionary}
Keep it open next to the manual (\href{@MANUAL@}{@MANUALCODE@}).
\begin{enumerate}[leftmargin=1.6em]
  \item \textbf{Look up a word.} The words are in A to Z order. The words at the top of each page
    are the first and the last word on that page, as in a printed dictionary.
  \item \textbf{Read what it means.} One line says what the word does, in the words of
    \code{sdata.help("word")}.
  \item \textbf{Try the example.} Every example runs as written. Most use a practice table, named
    in brackets after the example: \code{(marks)} means
    \code{table = sdata.table(sdata.example("marks"))} for a table word, or
    \code{chart = sdata.chart(sdata.example("marks"))} for a chart word.
  \item \textbf{Read the page.} \textbf{Manual p.} gives the page of the manual where the word is
    explained in full; the number is a link that opens the manual at that page. \textbf{Lesson}
    gives the lessons that teach it.
\end{enumerate}
Some words belong to more than one thing: \code{save} saves a chart, a table or numbers. Then
the word has one entry with a part for each, marked
\kind{chart}, \kind{table}, \kind{nums} and so on:
\begin{center}\small
\begin{tabular}{@{}ll@{}}
\kind{sdata} \code{sdata.chart(...)} & called on simplydata itself\\
\kind{table} \code{table.keep\_rows(...)} & a step on a table\\
\kind{chart} \code{chart.bars(...)} & a step on a chart\\
\kind{option} \code{show="median"} & an option inside a chart step\\
\kind{nums} \code{nums.average()} & a step on numbers: \code{nums = sdata.numbers([12, 15, 9])}\\
\kind{grid} \code{grid.row(1)} & a step on a grid: \code{grid = sdata.number\_grid([[3, 5], [8, 6]])}\\
\kind{dist} \code{dist.chance(above=60)} & a step on a distribution: \code{dist = sdata.distribution("normal")}\\
\end{tabular}
\end{center}
Know the name from pandas, R, NumPy or matplotlib (\code{groupby}, \code{iloc}, \code{hist},
\code{curve\_fit})? Look it up in \hyperref[other:names]{\textbf{Coming from another tool?}} at the end.

\section*{How the words are built}
\begin{itemize}[leftmargin=1.2em]
  \item \textbf{A word says what it does, in full English words}: \code{keep\_rows},
    \code{average\_of}, \code{compare\_averages}. No short forms to remember.
  \item \textbf{Words come in families.} Learn \code{keep\_rows} and you can guess \code{keep\_columns},
    \code{keep\_numbers} and \code{keep\_first\_rows}. The next page lists the families.
  \item \textbf{Steps on a table start with a verb} (\code{keep\_}, \code{remove\_}, \code{add\_},
    \code{make\_}, \code{fill\_}); \textbf{chart types are nouns} (\code{bars}, \code{dots}, \code{histogram}).
  \item \textbf{Anything for one axis ends in \code{\_x} or \code{\_y}}: \code{label\_y},
    \code{range\_x}, \code{log\_scale\_y}. x goes across, y goes up.
  \item \textbf{One option name means one thing everywhere}: \code{for\_each=} (a result for each group),
    \code{between=[low, high]}, \code{colour=}, \code{from\_column=}.
  \item \textbf{Counting starts at 1}, as people count: \code{nums.number\_at(1)} is the first number,
    \code{grid.row(1)} the top row.
  \item \textbf{Every change can be taken back} with \code{undo()}.
\end{itemize}
"""


# ---------------------------------------------------------------- reading the manual's page numbers
def _braced(text: str, i: int) -> tuple[str, int]:
    """The text inside the braces that open at text[i], and the index after the closing brace."""
    assert text[i] == "{", text[i:i + 20]
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    raise ValueError("unbalanced braces in the .aux file")


def read_aux(aux: Path, kinds: tuple[str, ...] = ("word", "lesson")) -> dict[str, tuple[str, str]]:
    """{label: (page, anchor)} for the labels of these kinds (word:..., lesson:...) in the manual's .aux file."""
    pages = {}
    text = aux.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"\\newlabel\{((?:" + "|".join(kinds) + r"):[^}]*)\}", text):
        body, _ = _braced(text, m.end())
        parts, i = [], 0
        while i < len(body) and len(parts) < 4:
            if body[i] == "{":
                part, i = _braced(body, i)
                parts.append(part)
            else:
                i += 1
        pages[m.group(1)] = (parts[1], parts[3])
    return pages


def manual_fingerprint() -> str:
    """A short fingerprint of the manual's .tex: the dictionary says which manual its pages are from."""
    text = make_manual.TEX.read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


_PAGE = re.compile(r"\\manualpage\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}")


def pages_in(tex: str) -> dict[str, tuple[str, str]]:
    """The page numbers written in a dictionary .tex: {label: (page, anchor)}."""
    return {label: (page, anchor) for label, page, anchor in _PAGE.findall(tex)}


def fingerprint_in(tex: str) -> str | None:
    m = re.search(r"^% Page numbers from simplydata_manual\.tex ([0-9a-f]+)$", tex, flags=re.M)
    return m.group(1) if m else None


# ---------------------------------------------------------------- the words
def headword(key: str) -> str:
    """The word as it is looked up: "table.keep_rows" -> "keep_rows", "show=" -> "show"."""
    return learn._word(key).rstrip("=")


def dict_label(word: str) -> str:
    return "dict:" + word.replace("_", "-")


def entries() -> dict[str, list[str]]:
    """{headword: [keys]}, A to Z; the keys of one headword in the order of KIND_ORDER."""
    found: dict[str, list[str]] = {}
    for key in learn._all_keys():
        found.setdefault(headword(key), []).append(key)
    for keys in found.values():
        keys.sort(key=lambda k: KIND_ORDER.index(learn._kind(k)))
    return {w: found[w] for w in sorted(found, key=sort_key)}


def sort_key(word: str) -> tuple:
    """Dictionary order: an underscore counts as a space (fit_line before fitdist), and a name
    written with its object in front (table.count) goes under its own first letter."""
    plain = word.rpartition(".")[2].lower().strip("=")
    return (plain.replace("_", " "), word)


def _calls(written: str) -> list[str]:
    """The words a pointer names: 'sdata.distribution("normal").chance(below=60)' -> ["dist.chance"]."""
    found = []
    for m in re.finditer(r"(?<![\w.)])(sdata|table|chart|nums|grid|dist)\.(\w+)\(", written):
        base = f"{m.group(1)}.{m.group(2)}"
        end = make_manual._balanced(written, m.end() - 1, "(", ")")
        name = base
        chained = re.match(r"\.(\w+)\(", written[end:]) if end else None
        if chained:
            if base != "sdata.distribution":
                raise ValueError(f"Say what {base}(...).{chained.group(1)} is in docs/make_dictionary.py")
            name = f"dist.{chained.group(1)}"
        key = learn._lookup(name)
        if key is None:
            raise ValueError(f"{name} in simplydata/_other_names.py is not a word")
        found.append(key)
    return list(dict.fromkeys(found))


def other_names() -> dict[str, list[str]]:
    """{name from another tool: [keys of the simplydata words it points to]}, A to Z.

    A name that is also a simplydata word is written with its object in front ("table.table",
    "sdata.grid"), so it is not mistaken for the simplydata word.
    """
    words = set(entries()) | {a for names in learn._aliases().values() for a in names}
    found: dict[str, list[str]] = {}
    for kind in ("sdata", "table", "chart", "numbers"):
        for name, written in OTHER_NAMES[kind].items():
            keys = _calls(written)
            assert keys, f"the other name {name} points to no word"
            shown = f"{WRITTEN[kind]}.{name}" if name in words else name
            found.setdefault(shown, [])
            found[shown] += [k for k in keys if k not in found[shown]]
    return {n: found[n] for n in sorted(found, key=sort_key)}


def other_names_of() -> dict[str, list[str]]:
    """{key: [names other tools use for it]}: the other names turned round."""
    turned: dict[str, list[str]] = {}
    for shown, keys in other_names().items():
        name = shown.rpartition(".")[2]
        for key in keys:
            if name not in turned.setdefault(key, []):
                turned[key].append(name)
    return {key: sorted(names, key=sort_key) for key, names in turned.items()}


def old_names() -> list[tuple[str, str | None, list[str]]]:
    """[(old name, new option name or None, [keys of the new word])], A to Z: renamed words (the
    keys say the new word) and renamed options (old= is now new= in the words of the keys)."""
    found: dict[tuple[str, str | None], list[str]] = {}
    olds = set()
    for cls, _ in learn._classes():
        olds |= set(getattr(cls, "_OLD_WORDS", {}))
    for old in sorted(olds):
        for name, _, key in learn._old_names(old):
            if not name.endswith("="):
                found.setdefault((name, None), []).append(key)
    for key in learn._all_keys():
        if key in learn.OPTIONS:
            continue
        for old, new in getattr(learn._function(key), "_renamed_options", {}).items():
            found.setdefault((f"{old}=", f"{new}="), []).append(key)
    rows = [(old, new, sorted(dict.fromkeys(keys), key=lambda k: KIND_ORDER.index(learn._kind(k))))
            for (old, new), keys in found.items()]
    return sorted(rows, key=lambda r: (sort_key(r[0]), r[1] or ""))


def lessons_of() -> dict[str, list[int]]:
    """{key: [numbers of the lessons that teach it]}, from each lesson's word list."""
    found: dict[str, list[int]] = {}
    for number, les in enumerate(learn.LESSONS, start=1):
        for word in les["words"]:
            key = learn._lookup(word)
            if key is not None and number not in found.setdefault(key, []):
                found[key].append(number)
    return found


def main_options(key: str) -> str:
    """The options in the brackets, in one line: needed ones plain, the others with =."""
    if key in learn.OPTIONS:
        return ""
    params = [p for p in inspect.signature(learn._function(key)).parameters.values()
              if p.name != "self" and p.kind is not p.VAR_KEYWORD
              and not (p.kind is p.VAR_POSITIONAL and p.name == "extra")]
    if not params:
        return "none"
    names = []
    for p in params:
        if p.kind is p.VAR_POSITIONAL:
            names.append(p.name + "...")
        elif p.default is p.empty:
            names.append(p.name)
        else:
            names.append(p.name + "=")
    shown = ", ".join(code(n) for n in names[:MAX_OPTIONS])
    if len(names) > MAX_OPTIONS:
        shown += f" and {len(names) - MAX_OPTIONS} more"
    return shown


def see_also(key: str) -> list[str]:
    """Related headwords: the other words of its group, the ones of the same family first."""
    own = headword(key)
    group = next(keys for _, keys in learn.GROUPS if key in keys)
    first = own.split("_")[0]
    here = group.index(key)
    others = [k for k in group if headword(k) != own]
    others.sort(key=lambda k: (headword(k).split("_")[0] != first, abs(group.index(k) - here)))
    return list(dict.fromkeys(headword(k) for k in others))[:MAX_SEE_ALSO]


# ---------------------------------------------------------------- LaTeX
def manual_page(label: str, pages: dict[str, tuple[str, str]]) -> str:
    if label not in pages:
        raise KeyError(f"The manual has no page for {label}: compile it first (python docs/make_manual.py --no-run)")
    page, anchor = pages[label]
    return f"\\manualpage{{{label}}}{{{page}}}{{{anchor}}}"


def word_link(word: str) -> str:
    return f"\\hyperref[{dict_label(word)}]{{{code(word)}}}"


def _sense(key: str, pages, turned, lessons) -> str:
    kind = learn._kind(key)
    out = [f"\\sense{{{WRITTEN[kind] if kind != 'option' else 'option'}}}{{{code(learn._call_form(key))}}}"]
    out.append(f"\\means{{{rich(learn._summary(key))}}}")
    example = code(learn._short_example(key))
    table = learn._example_table(key)
    if table:
        example += f" \\practice{{{esc(table)}}}"
    out.append(f"\\eg{{{example}}}")
    options = main_options(key)
    if options:
        out.append(f"\\field{{Options}}{{{options}}}")
    aliases = learn._aliases().get(key)
    if aliases:
        out.append(f"\\field{{Also written}}{{{', '.join(code(a) for a in aliases)}}}")
    if key in turned:
        out.append(f"\\field{{Other tools}}{{{', '.join(code(n) for n in turned[key])}}}")
    related = see_also(key)
    if related:
        out.append(f"\\field{{See also}}{{{', '.join(word_link(w) for w in related)}}}")
    where = "Manual " + manual_page(make_manual.label(key), pages)
    if key in lessons:
        where += "; " + ", ".join(f"lesson {n} ({manual_page(f'lesson:{n}', pages)})" for n in lessons[key])
    out.append(f"\\where{{{where}}}")
    return "\n".join(out)


def _letter(word: str) -> str:
    return word.rpartition(".")[2].lstrip("_")[0].upper()


def _words_part(pages) -> str:
    words = entries()
    turned = other_names_of()
    lessons = lessons_of()
    alias_of = {a: k for k, names in learn._aliases().items() for a in names}
    heads = {**{w: ("word", keys) for w, keys in words.items()},
             **{a: ("alias", k) for a, k in alias_of.items() if a not in words}}
    out = [r"\section*{The words, A to Z}\label{words}", r"\begin{dictcolumns}"]
    letter = None
    for word in sorted(heads, key=sort_key):
        if _letter(word) != letter:
            letter = _letter(word)
            out.append(f"\\letterhead{{{letter}}}")
        what, value = heads[word]
        if what == "alias":
            out.append(f"\\seeword{{{esc(word)}}}{{{word_link(headword(value))}}}")
            continue
        out.append(f"\\headword{{{esc(word)}}}{{{dict_label(word)}}}")
        out += [_sense(key, pages, turned, lessons) for key in value]
    out.append(r"\end{dictcolumns}")
    return "\n".join(out)


def _target(key: str, pages) -> str:
    """A word as it is written, linked to its entry, and its manual page."""
    return (f"\\hyperref[{dict_label(headword(key))}]{{{code(learn._call_form(key))}}}, "
            f"{manual_page(make_manual.label(key), pages)}")


def _others_part(pages) -> str:
    out = [r"\clearpage\section*{Coming from another tool?}\label{other:names}",
           "Names from pandas, Polars, R, NumPy, SciPy, matplotlib and seaborn, and everyday maths words. "
           "In simplydata they never run: " + code('sdata.help("groupby")') + " (or typing "
           + code("table.groupby()") + ") points to the simplydata word. Here each name points to the word "
           "and the manual page that explains it. A name that is also a simplydata word is written with "
           "its object in front, such as " + code("table.table") + ".",
           r"\begin{dictcolumns}"]
    letter = None
    for name, keys in other_names().items():
        if _letter(name) != letter:
            letter = _letter(name)
            out.append(f"\\letterhead{{{letter}}}")
        out.append(f"\\othername{{{esc(name)}}}{{{'; '.join(_target(k, pages) for k in keys)}}}")
    out.append(r"\end{dictcolumns}")
    out.append(r"\section*{Old names}\label{old:names}")
    out.append("Words and options that were renamed. The old name still works for one release, with a note "
               "that gives the new name.")
    out.append(r"\begin{dictcolumns}")
    for old, new, keys in old_names():
        where = "; ".join(_target(k, pages) for k in keys)
        if new:
            where = f"{code(new)} in {where}"
        out.append(f"\\othername{{{esc(old)}}}{{{where}}}")
    out.append(r"\end{dictcolumns}")
    return "\n".join(out)


def _families_part() -> str:
    words = list(entries())
    out = [r"\clearpage\section*{Word families}\label{families}\markboth{Word families}{Word families}",
           "A family is learned once and used again and again: the start or the end of a word says what "
           "kind of thing it does. Each family below lists its words; each word is a link to its entry.",
           r"\begin{familytable}"]
    for written, meaning, member in FAMILIES:
        members = [w for w in words if member(w)]
        assert members, f"the family {written} has no words"
        out.append(f"\\wordfamily{{{esc(written)}}}{{{esc(meaning)}}}{{{', '.join(word_link(w) for w in members)}}}")
    out.append(r"\end{familytable}")
    return "\n".join(out)


PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}
% no dates or random IDs in the PDF, so it only changes when the dictionary changes
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=-1
\usepackage[T1]{fontenc}
\usepackage{XCharter}
\usepackage[xcharter,vvarbb]{newtxmath}
\usepackage[lining,scale=0.88]{FiraSans}
\usepackage[varqu,varl,scaled=0.92]{zi4}
\usepackage{microtype}
\usepackage[a4paper,top=2.2cm,bottom=2.0cm,left=1.8cm,right=1.8cm,headheight=14pt,headsep=0.5cm]{geometry}
\usepackage[dvipsnames]{xcolor}
\usepackage{array}
\usepackage{longtable}
\usepackage{multicol}
\usepackage{enumitem}
\usepackage{needspace}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{bookmark}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.4em plus 0.2em}
\setlength{\emergencystretch}{3em}
\setlength{\columnsep}{1.4em}
\setlength{\columnseprule}{0.3pt}
\tolerance=3000
\raggedbottom

\definecolor{brand}{HTML}{1F5F99}
\definecolor{brandlight}{HTML}{E8F0F8}
\definecolor{codefg}{HTML}{24292F}
\definecolor{coderule}{HTML}{D5DCE4}
\definecolor{inkgrey}{HTML}{5A6570}
\colorlet{columnseprulecolor}{coderule}
\hypersetup{colorlinks=true,linkcolor=brand,urlcolor=brand,filecolor=brand,
  pdftitle={simplydata dictionary},pdfauthor={The simplydata project},
  pdfsubject={Every simplydata word, A to Z, with the page of the manual}}

\newcommand{\code}[1]{\texttt{\color{codefg}#1}}
\newenvironment{titleblock}{\par\noindent\begin{minipage}{\linewidth}}{\end{minipage}\par\vspace{0.6em}
  {\color{brand}\rule{\linewidth}{1.2pt}}\par}
\makeatletter
\renewcommand{\section}{\@startsection{section}{1}{0pt}{1.2em}{0.5em}{\sffamily\large\bfseries\color{brand}}}
\makeatother

% ---------------------------------------------------------------- entries
\newcommand{\kind}[1]{{\sffamily\scriptsize\bfseries\color{white}\colorbox{brand}{\strut #1}}}
\newcommand{\letterhead}[1]{\par\needspace{5\baselineskip}\vspace{0.6em}%
  \noindent{\sffamily\huge\bfseries\color{brand}#1}\par\nopagebreak\vspace{-0.3em}%
  \noindent{\color{brand}\rule{\linewidth}{0.8pt}}\par\nopagebreak}
\newcommand{\headword}[2]{\par\needspace{4\baselineskip}\vspace{0.35em}\phantomsection\label{#2}\markboth{#1}{#1}%
  \noindent{\sffamily\large\bfseries\color{brand}#1}\par\nopagebreak}
\newcommand{\seeword}[2]{\par\vspace{0.2em}\markboth{#1}{#1}\noindent{\sffamily\bfseries\color{brand}#1}\enspace
  {\small\itshape see} #2\par}
\newcommand{\sense}[2]{\par\nopagebreak\vspace{0.1em}\noindent\kind{#1}\enspace#2\par\nopagebreak}
\newcommand{\means}[1]{\noindent#1\par}
\newcommand{\eg}[1]{\noindent\hangindent=1em{\small\color{inkgrey}e.g.}\enspace{\small#1}\par}
\newenvironment{dictcolumns}{\begin{multicols}{2}\raggedright\setlength{\parskip}{0.15em}}{\end{multicols}}
\newcommand{\practice}[1]{{\small\color{inkgrey}(#1)}}
\newcommand{\field}[2]{\noindent\hangindent=1em{\sffamily\footnotesize\bfseries\color{inkgrey}#1}\enspace{\small#2}\par}
\newcommand{\where}[1]{\noindent\hangindent=1em{\sffamily\footnotesize\bfseries\color{brand}#1}\par}
\newcommand{\manualpage}[3]{\href{@MANUAL@\##3}{p.~#2}}
\newcommand{\othername}[2]{\par\markboth{#1}{#1}\noindent\hangindent=1em{\ttfamily\bfseries\color{codefg}#1}\enspace$\rightarrow$\enspace{\small#2}\par}
\newenvironment{familytable}{\par\small\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.2\linewidth}%
  >{\raggedright\arraybackslash}p{0.3\linewidth}>{\raggedright\arraybackslash}p{0.44\linewidth}@{}}
  \textbf{Family} & \textbf{What it does} & \textbf{Words}\\\hline\endhead}{\end{longtable}}
\newcommand{\wordfamily}[3]{{\ttfamily\bfseries\color{brand}#1} & #2 & #3\\[0.35em]}

% running heads: the first and the last word on the page, as in a printed dictionary
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\sffamily\small\bfseries\color{brand}\rightmark}
\fancyhead[C]{\sffamily\small\color{inkgrey}simplydata dictionary}
\fancyhead[R]{\sffamily\small\bfseries\color{brand}\leftmark}
\fancyfoot[C]{\sffamily\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\sffamily\small\thepage}\renewcommand{\headrulewidth}{0pt}}
"""

HEADER = """\
% ---------------------------------------------------------------------------------------------
% The simplydata dictionary. This file is made by docs/make_dictionary.py from simplydata's own
% code and the page numbers of the compiled manual. Do not edit it by hand: run
%     python docs/make_manual.py --no-run
% ---------------------------------------------------------------------------------------------
% Page numbers from simplydata_manual.tex @FINGERPRINT@"""


def make(pages: dict[str, tuple[str, str]], fingerprint: str) -> str:
    """The whole dictionary as LaTeX text, with the manual's page numbers in pages."""
    front = (FRONT.strip().replace("@VERSION@", esc(simplydata.__version__))
             .replace("@WORDS@", str(len(learn._all_keys())))
             .replace("@OTHERS@", str(len(other_names())))
             .replace("@MANUALCODE@", code(MANUAL_PDF_NAME)).replace("@MANUAL@", MANUAL_PDF_NAME))
    body = [
        HEADER.replace("@FINGERPRINT@", fingerprint),
        PREAMBLE.strip().replace("@MANUAL@", MANUAL_PDF_NAME),
        r"\begin{document}",
        r"\thispagestyle{plain}",
        front,
        _families_part(),
        r"\clearpage",
        _words_part(pages),
        _others_part(pages),
        r"\end{document}",
    ]
    return "\n\n".join(body) + "\n"


# ---------------------------------------------------------------- building
def build_folder() -> Path:
    return make_manual.latex_folder()


def compile_pdf(build: Path) -> None:
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
    for run in range(3):  # the running heads and links settle on the later runs
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", TEX.name],
            cwd=MANUAL, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            log = (build / f"{TEX.stem}.log").read_text(encoding="utf-8", errors="replace")
            print(log[-4000:])
            raise SystemExit(f"pdflatex failed on the dictionary (run {run + 1}); the full log is in {build}")
    shutil.copyfile(build / PDF.name, PDF)


def build(build: Path | None = None, pdf: bool = True) -> None:
    """Write the dictionary .tex from the compiled manual's page numbers, then compile it."""
    build = build or build_folder()
    aux = build / f"{make_manual.TEX.stem}.aux"
    if not aux.is_file():
        raise SystemExit(f"No {aux}: compile the manual first (python docs/make_manual.py --no-run)")
    TEX.write_text(make(read_aux(aux), manual_fingerprint()), encoding="utf-8", newline="\n")
    print(f"Wrote {TEX}")
    if pdf:
        print("Compiling the dictionary ...")
        compile_pdf(build)
        print(f"Wrote {PDF}")


if __name__ == "__main__":
    build(pdf="--tex" not in sys.argv[1:])
