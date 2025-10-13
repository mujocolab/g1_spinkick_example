"""Registers the custom spinkick task before running mjlab's play pipeline."""

import spinkick_example  # noqa: F401
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
