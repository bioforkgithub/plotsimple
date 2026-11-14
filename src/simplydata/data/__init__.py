"""Small made-up tables to practise with: sdata.example("marks").

Every number in these tables is invented. They are not real people, places or countries.
"""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

from .._errors import SimplydataError, suggest

# name -> (what it is, what each column means)
# Columns read as dates (the files keep them as year-month-day text).
DATE_COLUMNS = {"shop": ["date"]}
EXAMPLES = {
    "marks": (
        "Test marks of 18 made-up children in three classes.",
        {
            "name": "the child's first name",
            "class": "A, B or C",
            "subject": "Maths, Science or English",
            "hours": "hours spent studying for the test",
            "marks": "marks in the test, out of 100",
            "height": "the child's height in cm",
        },
    ),
    "weather": (
        "Weather in three made-up cities, month by month.",
        {
            "month": "Jan, Feb ... Dec",
            "month_number": "1 to 12",
            "city": "Sunnyvale, Rainbridge or Frostby",
            "temperature": "average temperature in °C",
            "rain": "rain that month in mm",
            "wind_direction": "where the wind mostly came from, in degrees (0 to 360)",
            "wind_speed": "average wind speed in km/h",
        },
    ),
    "fruit": (
        "Fruit sold by two made-up shops in one week. One row is one sale.",
        {
            "day": "Mon to Fri",
            "shop": "Corner shop or Market stall",
            "fruit": "apple, banana, cherry, grape, orange or pear",
            "count": "how many pieces of fruit were sold",
            "price": "price of one piece (made-up money)",
        },
    ),
    "growth": (
        "Population of four made-up countries, every two years from 2000 to 2024.",
        {
            "year": "2000, 2002 ... 2024",
            "country": "Applia, Bananaland, Cherria or Dateland",
            "population": "people, in millions",
            "forest": "how much of the land is forest, in %",
        },
    ),
    "shop": (
        "Daily sales of two made-up shops from January to March 2026. Some cells are empty.",
        {
            "date": "every day from 2026-01-01 to 2026-03-31 (real dates)",
            "shop": "Northgate or Southside",
            "sales": "money taken that day (made-up money); empty on days the till broke",
            "customers": "how many people bought something; empty on days nobody counted",
        },
    ),
    "school": (
        "Made-up pupils in 12 schools: study, sleep and test marks. Some sleep cells are empty.",
        {
            "school": "School A to School L (15 pupils in each)",
            "pupil": "P001 to P180",
            "hours": "hours spent studying for the test",
            "sleep": "hours of sleep the night before; empty when the pupil did not say",
            "breakfast": "yes or no: did the pupil eat breakfast that morning",
            "marks": "marks in the test, out of 100",
            "passed": "yes (50 marks or more) or no",
            "days_absent": "days the pupil missed school that term",
        },
    ),
    "meadow": (
        "Plants counted in 30 made-up plots of three fields, half of them grazed by sheep.",
        {
            "plot": "a short name for the plot, such as NG1",
            "field": "North field, South field or River field",
            "grazed": "yes or no: do sheep graze the plot",
            "clover": "how many clover plants in the plot",
            "daisy": "how many daisies",
            "buttercup": "how many buttercups",
            "dandelion": "how many dandelions",
            "nettle": "how many nettles",
            "thistle": "how many thistles",
            "moss": "how many patches of moss",
            "fern": "how many ferns",
            "orchid": "how many orchids",
            "yarrow": "how many yarrow plants",
        },
    ),
    "trial": (
        "120 made-up patients in a trial of a new medicine, followed for up to 3 years.",
        {
            "patient": "T001 to T120",
            "treatment": "new medicine or usual care",
            "age": "age in years when the patient joined",
            "days": "days from joining until the patient died or the follow-up ended",
            "died": "yes (died on that day) or no (still alive when the follow-up ended)",
        },
    ),
    "island": (
        "A map of a made-up island with two hills, on a 15 x 15 grid.",
        {
            "east": "how far east, 0 to 14 (km)",
            "north": "how far north, 0 to 14 (km)",
            "height": "height of the ground in m (below 0 is under the sea)",
        },
    ),
}


def _list_examples() -> None:
    print("Example tables you can use (all made up, for practice):\n")
    for name, (about, columns) in EXAMPLES.items():
        print(f'  sdata.example("{name}")')
        print(f"      {about}")
        print(f"      Columns: {', '.join(columns)}")
        print()
    print('Use one like this:  table = sdata.example("marks")  then  chart = sdata.chart(table)')


def example(name: str | None = None):
    """A small made-up table to practise with. sdata.example("marks")

    sdata.example() with no name lists every table and its columns.
    The tables are: marks, weather, fruit, growth, shop, school, meadow, trial, island.
    "shop" has empty cells and a column of dates to practise with; "school" has empty cells too.
    """
    if name is None:
        _list_examples()
        return None
    key = str(name).strip().lower()
    if key.endswith(".csv"):
        key = key[:-4]
    if key not in EXAMPLES:
        lines = [f'There is no example table called "{name}".']
        guess = suggest(key, EXAMPLES, n=1)
        if guess:
            lines.append(f'Did you mean sdata.example("{guess[0]}")?')
        lines.append("Tables you can use: " + ", ".join(EXAMPLES))
        raise SimplydataError("\n".join(lines))
    with (files(__name__) / f"{key}.csv").open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh, parse_dates=DATE_COLUMNS.get(key, False))
