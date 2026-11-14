"""Build the printable guide "30 days of simplydata" (docs/manual/simplydata_30_days.tex and .pdf).

The guide has the same days as sdata.today(): both are made from simplydata/learn_days.py. For each
of the two tracks (python and beginner) it has a one-page calendar to tick, then each day on its own
page (the warm-up questions with lines to write the answers on, what to do with the page of the
manual for each lesson, section and word, and the task), and the answers at the back.

The page numbers are read from the manual's .aux file, as for the dictionary, so the manual must be
compiled first. Building the manual builds the guide too:

    python docs/make_manual.py --no-run    # compile the manual, then the dictionary and this guide
    python docs/make_30_days.py            # only the guide, from the last compiled manual

A test (tests/test_learn_days.py) fails if the guide no longer matches simplydata/learn_days.py, or
if it was made from an older manual than docs/manual/simplydata_manual.tex.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_dictionary  # noqa: E402  (the same page-number helpers as the dictionary)
import make_manual  # noqa: E402
from make_manual import code, esc, listing, rich  # noqa: E402

import simplydata  # noqa: E402
from simplydata import learn, learn_days  # noqa: E402

MANUAL = make_manual.MANUAL
TEX = MANUAL / "simplydata_30_days.tex"
PDF = MANUAL / "simplydata_30_days.pdf"
MANUAL_PDF_NAME = make_manual.PDF.name
LABEL_KINDS = ("word", "lesson", "start", "ideas")

FRONT = r"""
\begin{titleblock}
{\fontsize{30}{34}\selectfont\sffamily\bfseries\color{brand} 30 days of simplydata\par}
\vspace{0.4em}
{\large\sffamily An hour a day: a warm-up from memory, a lesson, and a task to do yourself\par}
\vspace{0.3em}
{\sffamily\small\color{inkgrey} Version @VERSION@ \textbullet{} two tracks of 30 days \textbullet{} made from the code by
\code{docs/make\_30\_days.py}\par}
\end{titleblock}

\section*{How to use this guide}
The same 30 days are in Python. Start once, then type one word each day:
\begin{lstlisting}[style=py]
import simplydata as sdata
sdata.start_30_days()                    # the Python track: you already write some Python
sdata.start_30_days(track="beginner")    # the beginner track: you have never programmed
sdata.today()                            # every day: the day to do next
sdata.answers()                          # the answers to the warm-up and the task
\end{lstlisting}
This guide is the paper version, to keep next to the manual (\href{@MANUAL@}{@MANUALCODE@}): every
page number is a link that opens the manual at that page. Each day has three parts and takes about
an hour:
\begin{enumerate}[leftmargin=1.6em]
  \item \textbf{Warm-up, 5 minutes.} A few questions about words from earlier days. Answer them
    \emph{from memory}, on the lines, before you look anything up. Then check the answers at the
    back of the guide (each day has a link to them) or with \code{sdata.answers(n)}.
  \item \textbf{What to do, about 45 minutes.} The lesson to work through (\code{sdata.lesson(n)}),
    the pages of the manual to read and the words to look up with \code{sdata.help("word")}. Type every
    example yourself: typing it, and getting it wrong, teaches more than copying.
  \item \textbf{Try it yourself, 10 minutes.} A task to do without looking back. If yours runs and
    answers the question, it is right; the worked answers are at the back.
\end{enumerate}
Tick the boxes at the top of the page as you go, and the day on the calendar of your track.
\textbf{Missed a day, or a week?} That is fine. Carry on where you stopped: the days build on each
other, and nothing is lost by taking a break. In Python, \code{sdata.today()} does the same.

\section*{Why a warm-up from memory?}
Trying to remember something fixes it in memory better than reading it again: in experiments,
people who were tested on a text remembered more of it days later than people who read it again
\citeref{Roediger and Karpicke 2006}. Practice spread over several days also works better than the
same practice in one go \citeref{Cepeda et al.\ 2006}. So each warm-up asks one question about
yesterday, one about three days ago and one about a week ago (and, later on, two weeks ago). Review
days (@REVIEW@) ask about the whole course, and the questions change each time a day comes back.

\section*{Which track?}
\begin{center}\small
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.17\linewidth}>{\raggedright\arraybackslash}p{0.75\linewidth}@{}}
@TRACKS@
\end{tabular}
\end{center}
"""

REFERENCES = r"""
\section*{References}
{\small
Roediger, H.~L., \& Karpicke, J.~D. (2006). Test-enhanced learning: taking memory tests improves long-term
retention. \emph{Psychological Science}, 17(3), 249--255. \url{https://doi.org/10.1111/j.1467-9280.2006.01693.x}

Cepeda, N.~J., Pashler, H., Vul, E., Wixted, J.~T., \& Rohrer, D. (2006). Distributed practice in verbal recall
tasks: a review and quantitative synthesis. \emph{Psychological Bulletin}, 132(3), 354--380.
\url{https://doi.org/10.1037/0033-2909.132.3.354}\par}
"""

PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}
% no dates or random IDs in the PDF, so it only changes when the guide changes
\pdfinfoomitdate=1
\pdftrailerid{}
\pdfsuppressptexinfo=-1
\usepackage[T1]{fontenc}
\usepackage{XCharter}
\usepackage[xcharter,vvarbb]{newtxmath}
\usepackage[lining,scale=0.88]{FiraSans}
\usepackage[varqu,varl,scaled=0.92]{zi4}
\usepackage{microtype}
\usepackage[a4paper,top=2.0cm,bottom=1.9cm,left=1.9cm,right=1.9cm,headheight=14pt,headsep=0.5cm]{geometry}
\usepackage[dvipsnames]{xcolor}
\usepackage{array}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{needspace}
\usepackage{fancyhdr}
\usepackage{listings}
\usepackage[most]{tcolorbox}
\usepackage[hidelinks]{hyperref}
\usepackage{bookmark}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em plus 0.2em}
\setlength{\emergencystretch}{3em}
\tolerance=3000
\raggedbottom

% the colours of the manual and the dictionary
\definecolor{brand}{HTML}{1F5F99}
\definecolor{brandlight}{HTML}{E8F0F8}
\definecolor{codebg}{HTML}{F5F7FA}
\definecolor{coderule}{HTML}{D5DCE4}
\definecolor{codefg}{HTML}{24292F}
\definecolor{codekw}{HTML}{1F5F99}
\definecolor{codestr}{HTML}{A33B00}
\definecolor{codecom}{HTML}{6A737D}
\definecolor{goodbg}{HTML}{EEF7EE}
\definecolor{goodframe}{HTML}{2E7D32}
\definecolor{trybg}{HTML}{FFF4E5}
\definecolor{tryframe}{HTML}{C25E00}
\definecolor{inkgrey}{HTML}{5A6570}
\hypersetup{colorlinks=true,linkcolor=brand,urlcolor=brand,filecolor=brand,
  pdftitle={30 days of simplydata},pdfauthor={The simplydata project},
  pdfsubject={An hour a day for 30 days: warm-up questions, lessons and tasks}}

\newcommand{\code}[1]{\texttt{\color{codefg}#1}}
\lstset{
  basicstyle=\ttfamily\small\color{codefg},
  columns=fullflexible, keepspaces=true, upquote=true, showstringspaces=false,
  breaklines=true, breakatwhitespace=false,
  postbreak=\mbox{\textcolor{codecom}{$\hookrightarrow$}\space},
}
\lstdefinestyle{py}{
  language=Python,
  keywordstyle=\color{codekw}\bfseries, stringstyle=\color{codestr}, commentstyle=\color{codecom},
  emph={sdata}, emphstyle=\color{codekw},
  backgroundcolor=\color{codebg}, frame=single, rulecolor=\color{coderule}, framerule=0.4pt,
  framesep=4pt, xleftmargin=2.0em, xrightmargin=4pt,
  numbers=left, numberstyle=\tiny\sffamily\color{codecom}, numbersep=10pt,
  aboveskip=0.5em, belowskip=0.4em,
}
\newenvironment{titleblock}{\par\noindent\begin{minipage}{\linewidth}}{\end{minipage}\par\vspace{0.6em}
  {\color{brand}\rule{\linewidth}{1.2pt}}\par}
\makeatletter
\renewcommand{\section}{\@startsection{section}{1}{0pt}{1.1em}{0.4em}{\sffamily\large\bfseries\color{brand}}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{0pt}{0.9em}{0.3em}{\sffamily\bfseries\color{brand}}}
\makeatother
\newcommand{\citeref}[1]{{\small(#1)}}
\setlist[itemize]{itemsep=0.15em,parsep=0pt,topsep=0.2em,leftmargin=1.3em}
\setlist[enumerate]{itemsep=0.15em,parsep=0pt,topsep=0.2em,leftmargin=1.6em}

% ---------------------------------------------------------------- word kinds, as in the dictionary
\newcommand{\kind}[1]{{\sffamily\scriptsize\bfseries\color{white}\colorbox{brand}{\strut #1}}}
\newcommand{\manualpage}[3]{\href{@MANUAL@\##3}{p.~#2}}

% ---------------------------------------------------------------- a day
\newcommand{\tick}{\raisebox{-0.1em}{\Large$\square$}}
\newcommand{\daystart}[4]{\clearpage\phantomsection\label{#4}\markboth{#3}{Day #1}%
  \addcontentsline{toc}{subsection}{Day #1: #2}%
  \noindent\begin{minipage}[b]{0.13\linewidth}{\sffamily\fontsize{30}{30}\selectfont\bfseries\color{brand}#1}\end{minipage}%
  \begin{minipage}[b]{0.87\linewidth}{\sffamily\small\color{inkgrey}Day #1 of 30 \textbullet{} #3}\par
  {\sffamily\Large\bfseries\color{brand}#2}\end{minipage}\par\vspace{0.2em}
  {\color{brand}\rule{\linewidth}{0.8pt}}\par\vspace{0.1em}
  {\sffamily\small\tick\ Warm-up\qquad\tick\ What to do\qquad\tick\ Task\qquad Date:\ \rule{3cm}{0.4pt}}\par}
\newcommand{\goal}[1]{\par\noindent{\sffamily\bfseries\color{brand}Goal.}\enspace#1\par}
\tcbset{boxbase/.style={enhanced,breakable,boxrule=0pt,leftrule=3pt,arc=1pt,outer arc=1pt,
  left=7pt,right=7pt,top=5pt,bottom=5pt,before skip=0.6em,after skip=0.6em,
  fonttitle=\sffamily\bfseries,coltitle=black,attach title to upper,title after break={}}}
\newtcolorbox{warmup}{boxbase,colback=brandlight,colframe=brand,
  title={\color{brand}1\enspace Warm-up, 5 minutes: from memory, without looking back\par}}
\newtcolorbox{todo}{boxbase,colback=white,colframe=brand,
  title={\color{brand}2\enspace What to do\par}}
\newtcolorbox{tryit}{boxbase,colback=trybg,colframe=tryframe,
  title={\color{tryframe}3\enspace Try it yourself, without looking back\par}}
\newtcolorbox{answerbox}{boxbase,colback=goodbg,colframe=goodframe,before skip=0.3em,after skip=0.5em}
\newcommand{\fromday}[1]{{\footnotesize\sffamily\color{inkgrey}(day~#1)}}
\newcommand{\writeline}{\par\vspace{0.9em}\noindent{\color{coderule}\rule{\linewidth}{0.4pt}}\par}
\newcommand{\field}[2]{\par\noindent\hangindent=1em{\sffamily\small\bfseries\color{inkgrey}#1}\enspace{\small#2}\par}
\newcommand{\whateachline}{\par\noindent{\sffamily\small\bfseries\color{brand}What each line does}\par}
\newlist{lineexplain}{description}{1}
\setlist[lineexplain]{leftmargin=2.0em,labelwidth=1.4em,labelsep=0.6em,align=right,
  font=\normalfont\sffamily\footnotesize\color{codecom},itemsep=0.1em,parsep=0pt,topsep=0.2em}

% ---------------------------------------------------------------- calendar and answers
\newcommand{\trackstart}[3]{\clearpage\phantomsection\label{#3}\markboth{#1}{#1}%
  \addcontentsline{toc}{section}{#1}%
  {\sffamily\fontsize{24}{28}\selectfont\bfseries\color{brand}#1\par}\vspace{0.2em}
  {\sffamily\large\color{inkgrey}#2\par}\vspace{0.2em}{\color{brand}\rule{\linewidth}{1.2pt}}\par}
\newcommand{\answerday}[3]{\par\needspace{6\baselineskip}\vspace{0.6em}\phantomsection\label{#3}%
  \noindent{\sffamily\bfseries\color{brand}Day #1: #2}\par\nopagebreak}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\sffamily\small\color{inkgrey}30 days of simplydata}
\fancyhead[R]{\sffamily\small\bfseries\color{brand}\leftmark}
\fancyfoot[C]{\sffamily\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\sffamily\small\thepage}\renewcommand{\headrulewidth}{0pt}}
"""

HEADER = """\
% ---------------------------------------------------------------------------------------------
% 30 days of simplydata: the printable guide. This file is made by docs/make_30_days.py from
% simplydata/learn_days.py and the page numbers of the compiled manual. Do not edit it by hand:
% change simplydata/learn_days.py, then run
%     python docs/make_manual.py --no-run
% ---------------------------------------------------------------------------------------------
% Page numbers from simplydata_manual.tex @FINGERPRINT@"""


# ---------------------------------------------------------------- labels
def day_label(track: str, number: int) -> str:
    return f"day:{track}:{number}"


def answers_label(track: str, number: int) -> str:
    return f"answers:{track}:{number}"


def manual_page(label: str, pages: dict[str, tuple[str, str]]) -> str:
    return make_dictionary.manual_page(label, pages)


def word_ref(key: str, pages) -> str:
    """A word as it is written, with its kind and the manual page that explains it."""
    kind = learn._kind(key)
    badge = make_dictionary.WRITTEN[kind] if kind != "option" else "option"
    return (f"\\kind{{{badge}}}\\,{code(learn._call_form(key))}~"
            f"{manual_page(make_manual.label(key), pages)}")


# ---------------------------------------------------------------- parts of a day
def _warm_up(p: dict) -> str:
    out = [r"\begin{warmup}"]
    if not p["warm_up"]:
        out.append("Nothing to remember yet on day 1: just start.")
    else:
        out.append(r"\begin{enumerate}")
        for question, _, source in p["warm_up"]:
            out.append(f"  \\item {rich(question)} \\fromday{{{source}}}\\writeline")
        out.append(r"\end{enumerate}")
        out.append(f"{{\\small Answers: \\hyperref[{answers_label(p['track'], p['number'])}]"
                   f"{{page~\\pageref*{{{answers_label(p['track'], p['number'])}}}}}, "
                   f"or {code('sdata.answers(' + str(p['number']) + ')')}.}}")
    out.append(r"\end{warmup}")
    return "\n".join(out)


def _todo(p: dict, pages) -> str:
    out = [r"\begin{todo}", r"\begin{itemize}"]
    out += [f"  \\item {rich(line)}" for line in p["do"]]
    out.append(r"\end{itemize}")
    refs = [f"lesson~{n}, {manual_page(f'lesson:{n}', pages)}" for n in p["lessons"]]
    refs += [f"``{esc(title)}'', {manual_page(label, pages)}" for label, title in p["manual"]]
    if refs:
        out.append(f"\\field{{In the manual}}{{{'; '.join(refs)}.}}")
    if p["words"]:
        out.append(f"\\field{{Words to look up}}{{{'; '.join(word_ref(k, pages) for k in p['words'])}.}}")
    for block in p["examples"]:
        out.append(r"\par\noindent{\sffamily\small\bfseries\color{brand}Example: type it yourself}\par")
        out.append(listing("\n".join(c for c, _ in block)))
        out.append(r"\whateachline")
        out.append(r"\begin{lineexplain}")
        out += [f"  \\item[{i}] {rich(explain)}" for i, (_, explain) in enumerate(block, start=1)]
        out.append(r"\end{lineexplain}")
    if p["errors"]:
        out.append(r"\par\noindent{\sffamily\small\bfseries\color{brand}Errors to make on purpose} "
                   r"{\small(read the last line of each error first)}\par")
        for source, _, meaning in p["errors"]:
            out.append(listing(source))
            out.append(r"\vspace{-0.3em}{\small " + rich(meaning) + r"}\par")
    out.append(r"\end{todo}")
    return "\n".join(out)


def _tryit(p: dict) -> str:
    out = [r"\begin{tryit}", r"\begin{enumerate}"]
    out += [f"  \\item {rich(task)}" for task in p["task"]]
    out.append(r"\end{enumerate}")
    label = answers_label(p["track"], p["number"])
    out.append(f"{{\\small Worked answers: \\hyperref[{label}]{{page~\\pageref*{{{label}}}}}, or "
               f"{code('sdata.answers(' + str(p['number']) + ')')}.}}")
    out.append(r"\end{tryit}")
    return "\n".join(out)


def _day(track: str, number: int, pages) -> str:
    p = learn_days.plan(track, number)
    title = learn_days.TRACKS[track]["title"]
    return "\n".join([
        f"\\daystart{{{number}}}{{{esc(p['title'])}}}{{{esc(title)}}}{{{day_label(track, number)}}}",
        f"\\goal{{{rich(p['goal'])} About an hour.}}",
        _warm_up(p),
        _todo(p, pages),
        _tryit(p),
    ])


# ---------------------------------------------------------------- calendar and answers
_KIND_NAME = {"review": "review", "project": "project", "final": "final test", "learn": ""}


def _calendar(track: str) -> str:
    info = learn_days.TRACKS[track]
    out = [f"\\trackstart{{{esc(info['title'])}}}{{{esc(info['who'][0].upper() + info['who'][1:])}}}{{track:{track}}}",
           rich(info["about"]) + " Tick each day when you have done it; the day's number is a link to its page.",
           r"\begin{center}\small",
           r"\begin{longtable}{@{}c r >{\raggedright\arraybackslash}p{0.56\linewidth} >{\raggedright\arraybackslash}p{0.14\linewidth} r@{}}",
           r"\toprule & \textbf{Day} & \textbf{What you do} & \textbf{Lesson} & \textbf{Page}\\\midrule\endhead"]
    for number, (title, _, units, kind) in enumerate(learn_days.DAYS[track], start=1):
        lessons = sorted({n for u in units for n in learn_days.UNITS[u]["lessons"]})
        lesson_text = (", ".join(map(str, lessons)) or "-") if kind == "learn" else _KIND_NAME[kind]
        label = day_label(track, number)
        out.append(f"\\tick & \\hyperref[{label}]{{{number}}} & {esc(title)} & {esc(lesson_text)} & "
                   f"\\pageref*{{{label}}}\\\\[0.25em]")
    out.append(r"\bottomrule\end{longtable}\end{center}")
    return "\n".join(out)


def _answers(track: str) -> str:
    title = learn_days.TRACKS[track]["title"]
    out = [f"\\trackstart{{Answers: {esc(title)}}}{{The warm-up answers and a worked answer to each task}}"
           f"{{answers:{track}}}",
           "There is often more than one good answer: if yours runs and answers the question, it is right. "
           "Every piece of code here runs as it is (the tests run it)."]
    for number in range(1, learn_days.DAYS_IN_A_TRACK + 1):
        p = learn_days.plan(track, number)
        out.append(f"\\answerday{{{number}}}{{{esc(p['title'])}}}{{{answers_label(track, number)}}}")
        if p["warm_up"]:
            out.append(r"\begin{enumerate}")
            for question, answer, source in p["warm_up"]:
                out.append(f"  \\item {rich(question)} \\fromday{{{source}}}\\\\ $\\rightarrow$ {rich(answer)}")
            out.append(r"\end{enumerate}")
        for i, (task, (source, explain)) in enumerate(zip(p["task"], p["solutions"]), start=1):
            out.append(f"\\par\\noindent{{\\sffamily\\small\\bfseries\\color{{tryframe}}Task {i}.}} {rich(task)}\\par")
            if source:
                out.append(listing(source))
            out.append(f"\\begin{{answerbox}}{rich(explain)}\\end{{answerbox}}")
        out.append(f"{{\\small\\hyperref[{day_label(track, number)}]{{Back to day {number}}}}}")
    return "\n".join(out)


def _tracks_table() -> str:
    rows = []
    for track in learn_days.TRACK_NAMES:
        info = learn_days.TRACKS[track]
        start = 'sdata.start_30_days()' if track == "python" else f'sdata.start_30_days(track="{track}")'
        rows.append(f"\\hyperref[track:{track}]{{\\textbf{{{esc(info['title'])}}}}} & "
                    f"{esc(info['who'][0].upper() + info['who'][1:])}. {rich(info['about'])} Start: {code(start)}."
                    f"\\\\[0.4em]")
    return "\n".join(rows)


def _words_table(pages) -> str:
    keys = next(keys for group, keys in learn.GROUPS if group == "30 days of simplydata")
    rows = [r"\section*{The words in Python}", r"\begin{center}\small",
            r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.36\linewidth}>{\raggedright\arraybackslash}p{0.58\linewidth}@{}}"]
    for key in keys:
        rows.append(f"{word_ref(key, pages)} & {rich(learn._summary(key))}\\\\[0.3em]")
    rows.append(r"\end{tabular}\end{center}")
    rows.append("Your progress is kept only on your computer, in a small file (" + code("~/.simplydata/progress.json")
                + "). Nothing is sent anywhere.")
    return "\n".join(rows)


def _contents() -> str:
    rows = [r"\section*{In this guide}", r"\begin{center}\small",
            r"\begin{tabular}{@{}l r@{}}"]
    for track in learn_days.TRACK_NAMES:
        title = esc(learn_days.TRACKS[track]["title"])
        rows.append(f"\\hyperref[track:{track}]{{{title}: the calendar and the 30 days}} & "
                    f"page~\\pageref*{{track:{track}}}\\\\")
    for track in learn_days.TRACK_NAMES:
        title = esc(learn_days.TRACKS[track]["title"])
        rows.append(f"\\hyperref[answers:{track}]{{Answers: {title}}} & page~\\pageref*{{answers:{track}}}\\\\")
    rows.append(r"\end{tabular}\end{center}")
    return "\n".join(rows)


def make(pages: dict[str, tuple[str, str]], fingerprint: str) -> str:
    """The whole guide as LaTeX text, with the manual's page numbers in pages."""
    reviews = sorted({n for t in learn_days.TRACK_NAMES for n, d in enumerate(learn_days.DAYS[t], start=1)
                      if d[3] == "review"})
    review_text = ", ".join(map(str, reviews[:-1])) + " and " + str(reviews[-1])
    front = (FRONT.strip().replace("@VERSION@", esc(simplydata.__version__))
             .replace("@MANUALCODE@", code(MANUAL_PDF_NAME)).replace("@MANUAL@", MANUAL_PDF_NAME)
             .replace("@REVIEW@", review_text).replace("@TRACKS@", _tracks_table()))
    body = [
        HEADER.replace("@FINGERPRINT@", fingerprint),
        PREAMBLE.strip().replace("@MANUAL@", MANUAL_PDF_NAME),
        r"\begin{document}",
        r"\thispagestyle{plain}",
        front,
        _words_table(pages),
        _contents(),
        REFERENCES.strip(),
    ]
    for track in learn_days.TRACK_NAMES:
        body.append(_calendar(track))
        body += [_day(track, n, pages) for n in range(1, learn_days.DAYS_IN_A_TRACK + 1)]
    for track in learn_days.TRACK_NAMES:
        body.append(_answers(track))
    body.append(r"\end{document}")
    return "\n\n".join(body) + "\n"


# ---------------------------------------------------------------- building
def compile_pdf(build: Path) -> None:
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
    for run in range(3):  # the page references settle on the later runs
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}", TEX.name],
            cwd=MANUAL, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            log = (build / f"{TEX.stem}.log").read_text(encoding="utf-8", errors="replace")
            print(log[-4000:])
            raise SystemExit(f"pdflatex failed on the 30-day guide (run {run + 1}); the full log is in {build}")
    shutil.copyfile(build / PDF.name, PDF)


def build(build: Path | None = None, pdf: bool = True) -> None:
    """Write the guide's .tex from the compiled manual's page numbers, then compile it."""
    build = build or make_manual.latex_folder()
    aux = build / f"{make_manual.TEX.stem}.aux"
    if not aux.is_file():
        raise SystemExit(f"No {aux}: compile the manual first (python docs/make_manual.py --no-run)")
    pages = make_dictionary.read_aux(aux, kinds=LABEL_KINDS)
    TEX.write_text(make(pages, make_dictionary.manual_fingerprint()), encoding="utf-8", newline="\n")
    print(f"Wrote {TEX}")
    if pdf:
        print("Compiling the 30-day guide ...")
        compile_pdf(build)
        print(f"Wrote {PDF}")


if __name__ == "__main__":
    build(pdf="--tex" not in sys.argv[1:])
