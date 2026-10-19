# Prompt: add functional notebooks to a project

Copy everything below the line into a new Claude Code session in the target
project. It describes the idea as it works in ARIEL (`tests/functional/*.ipynb`)
and what to build elsewhere.

---

Add a set of **functional notebooks** to this project, the way ARIEL has them.
Read this whole prompt first, then look at the project, then ask me questions
before you write anything.

## What a functional notebook is

A Jupyter notebook in `tests/functional/` that shows one feature area of the
project working end to end, with output a person can judge in a few seconds.
It is three things at once:

1. A tutorial. Someone new opens it and learns what the feature does and how to
   call it, from top to bottom, without reading the source.
2. A visual check. Where the result can be seen (a picture, a plot, a table),
   it is shown inline, so a wrong result is obvious to the eye. Unit tests say
   a number is right; the notebook says the thing looks right.
3. A smoke test. It runs top to bottom with one command, unattended, on a clean
   checkout, and fails loudly if any cell raises.

It is not a replacement for unit tests. Unit tests pin exact behaviour and
carry the coverage number. Notebooks cover what a unit test cannot say well:
"does the rendered robot look like a robot", "do the environments all load and
look different", "is the speed in a believable range".

## Conventions to copy

- **Location and names.** `tests/functional/<topic>.ipynb`, one notebook per
  feature area, not per module. Examples from ARIEL: `sim_environments`
  (every simulated world, rendered in a grid), `decoders` (the three ways to
  turn numbers into a robot body, each drawn), `cpg` (a controller driven and
  plotted), `nde` (an encoding, sampled and measured), `speed_tests` (timing
  tables).
- **Cell layout.** Cell 1 is a markdown title and two or three sentences:
  what this area is and what the reader will see. Then imports in separate
  cells, grouped under a comment (`# Standard library`, `# Third-party
  libraries`, `# Local libraries`). Then constants (`SEED`, a random generator
  built from it, an output folder). Then one `##` section per behaviour:
  a short markdown cell saying what is about to happen, then small code cells,
  each doing one thing and ending in something visible.
- **Fixed seeds.** Every random source starts from a named seed, so two runs
  give the same picture.
- **Visible output.** Prefer an inline picture (`matplotlib`, a rendered frame
  shown with `plt.imshow`), a table (`rich.table.Table` or a pandas frame), or
  a short printed summary. Print the values that matter, not everything.
- **Cheap assertions.** Close each section with one or two plain `assert`
  lines on invariants (a count, a shape, a range, "output is not all zeros").
  They turn a notebook that only shows into one that also fails.
- **Headless.** No window pops up. The notebook must run on a server with no
  display; if a feature needs a graphics backend, say so in the first cell and
  skip with a clear message when it is missing.
- **Writes only to a scratch folder** such as `__data__/` (git-ignored). Never
  into the source tree, never into the user's real data.
- **Outputs are committed.** Save the notebook with its outputs so it can be
  read on GitHub as a tutorial without running anything. Keep them small
  (downscale pictures; avoid hundreds of KB of base64).
- **Formatting.** Format the code cells with the project's formatter (in
  ARIEL, `ruff format` also formats notebook cell source). Keep the cell source
  free of paths that exist only on one machine.

## What to build

1. **A runner.** One command that executes every notebook in
   `tests/functional/` from a clean kernel, with a timeout per notebook, and
   prints one line each: name, pass or fail, seconds. A failing cell prints the
   cell number and the error. Exit code is non-zero if any notebook fails. Use
   the lightest tool that does this (for example `nbclient` or
   `jupyter nbconvert --execute`); do **not** add a dependency without telling
   me its name, source and size first. Add the command to the project's
   CLAUDE.md or README commands table.
2. **A coverage table.** A short markdown file next to the notebooks
   (`tests/functional/README.md`) with one row per public package or feature
   area of the project: the notebook that covers it, or "none yet". The rule
   to hold afterwards: every public feature area has a row with a notebook.
3. **The notebooks themselves**, starting with the feature areas a newcomer
   meets first, in the conventions above.
4. **Keeping them alive.** Say in the README how often to run them (before a
   release, after changing the area they cover) and what a failure means. If
   the project has CI, propose a job, but do not add one without asking.

## How to work

- First read the project: its README, CLAUDE.md or contributing notes, the
  package layout, how tests and the formatter are run, and whether Jupyter
  tooling is already a dependency. Then **ask me questions** (at most four at
  a time, recommended option first): which feature areas get a notebook and in
  what order, which runner you propose and what it costs to add, and what
  should happen where a notebook needs hardware or a service that is not
  always there.
- Do the work one notebook per commit, run it before you commit it, and tell
  me what you ran and what you could not run.
- Match the project's own style for code, comments and docs. Where it has no
  rule, keep notebooks plain: short cells, no clever helpers, no hidden state
  between sections that a reader cannot see.
- Do not change any production code to make a notebook pass. If a notebook
  exposes a bug, record it as a finding and ask me.
