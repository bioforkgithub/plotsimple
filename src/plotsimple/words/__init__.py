"""The plotsimple vocabulary, one file per group of words.

Each file defines a Words class (a mixin). The Chart class in plotsimple/chart.py
inherits them all. To add a word: add a method to the right file, decorate it with
@friendly, check input straight away, and record a step with self._add(...).
"""
