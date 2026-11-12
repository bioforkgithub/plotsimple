"""Numbers: plain-English words over NumPy for lists and grids of numbers.

    nums = numbers([3, 5, 2, 8])
    nums.add(5)            # words change the numbers (nums.undo() takes it back)
    bigger = nums * 2      # symbols make new numbers
    nums.average()

ps.lesson(13) teaches them.
"""

from .numbers import NumberGrid, Numbers, number_grid, numbers, random_numbers

__all__ = ["NumberGrid", "Numbers", "number_grid", "numbers", "random_numbers"]
