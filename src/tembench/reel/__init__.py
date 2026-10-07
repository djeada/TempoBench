"""Short vertical videos that retell a benchmark: measure, fit, verdict.

`build_story` works out what to show from a summary (and the run's trial
records); `render_reel` draws it.  Rendering needs matplotlib (the `reel`
extra) and ffmpeg for video; the story itself does not.
"""

from .story import Candidate, Series, Story, Trial, build_story, pretty_model

__all__ = ["Candidate", "Series", "Story", "Trial", "build_story", "pretty_model"]
