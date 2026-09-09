"""Build the shipped demo set into ``demo/``, from the sample data in
``sample data/``.

Run directly, from the repository root::

    python _make_demo_project.py

writes the whole set (the "Getting started" project first, then one file per
subject) into ``demo/``. The application's "Load demo" reads pre-built
files from there rather than building them on the spot - see
``app.data.demo_project.copy_demo_project`` - so this needs to be run once
after cloning, and again whenever the demo set itself changes.

Pass ``--output PATH`` to write a single ``.dhub`` file instead (the complete
project by default), or ``--all DIRECTORY`` to write the whole set somewhere
other than ``demo/``. All of the actual generation logic lives in
``app/data/demo_project.py``, which is also what ``app/tests/test_demo_projects.py``
imports - this script is only the command line entry point named for what it
does.
"""
from __future__ import annotations

from app.data.demo_project import main

if __name__ == "__main__":
    main()
