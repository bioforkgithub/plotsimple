# Contributing to simplydata

Thank you for helping. simplydata is a dictionary of simple words over pandas, Polars, NumPy,
SciPy, matplotlib and seaborn. The words are what matter most, so every contribution, big or
small, is welcome.

## Suggest a word or report a problem

Open an [issue](https://github.com/bioforkgithub/simplydata/issues). For a problem, include the
few lines of code that show it, what you expected and what happened. For a new word, say what
it should do and how you would write it.

## Rules for words

A new word should pass these checks before it is added:

1. **Read it aloud.** `table.keep_rows("marks", above=50)` should sound like a short sentence.
2. **The word names the idea.** `compare_averages`, not `ttest`. Keep mathematical words as
   they are (log, median, standard deviation).
3. **Guess it from its sibling.** Reuse the place words (`top`, `bottom`, `x`, `y`, `at`,
   `between`) and the patterns that already exist (`keep_rows`, `keep_first_rows`).
4. **Simple words, full power.** Simplicity is in the words, never a limit on what the
   libraries underneath can do.
5. **Say what was done.** Results name the method used (for example "Welch's t-test"), and
   empty cells are never dropped silently.

## Add code

```
git clone https://github.com/bioforkgithub/simplydata
cd simplydata
python -m venv venv            # or keep the venv outside the project folder
venv/bin/pip install -e ".[test]"      # Windows: venv\Scripts\pip install -e ".[test]"
python -m pytest -q
```

- Each new word needs tests. Statistical results are checked against a second tool (R, SciPy
  or statsmodels) with the reference values written in the test.
- Each new word needs an entry in `src/simplydata/learn.py` (for `sdata.words()` and
  `sdata.help()`); the tests fail until it has one.
- If you change words or lessons, rebuild the cheat sheet with `python docs/make_cheatsheet.py`
  and, if you have LaTeX, the manual with `python docs/make_manual.py`.
- Tests must pass on Python 3.10 to 3.13; GitHub runs them on Windows, macOS and Linux for
  every pull request.

By contributing you agree that your work is shared under the MIT licence.
