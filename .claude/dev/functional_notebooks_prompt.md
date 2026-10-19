# Prompt: add functional notebooks to a project

Copy everything below the line into a new Claude Code session in the target
project. It is the general version of the idea as it works in ARIEL
(`tests/functional/*.ipynb`); updated 2026-10-19 with slow notebooks, file
arguments, SKIP reasons and pre-commit hooks.

---

# add functional notebooks to a project

Add a set of functional notebooks to this project. Read this whole prompt first, then look at the project, then ask me questions before you write anything.

## What a functional notebook is

A Jupyter notebook that shows one feature area of the project working end to end, with output a person can judge in a few seconds. It is three things at once:

1. A tutorial. Someone new opens it and learns what the feature does and how to call it, top to bottom, without reading the source.
2. A visual check. Where the result can be seen (a picture, a plot, a table), it is shown inline, so a wrong result is obvious to the eye. Unit tests say a number is right; the notebook says the thing looks right.
3. A smoke test. It runs top to bottom with one command, unattended, on a clean checkout, and fails loudly if any cell raises.

It does not replace unit tests. Unit tests pin exact behaviour and carry the coverage number. Notebooks cover what a unit test says badly: "does the rendered output look right", "do all the variants load and look different", "is the speed in a believable range".

## Conventions

- **Location and names.** `tests/functional/<topic>.ipynb`, one notebook per feature area, not per module. Name it after what the reader will see (`rendering`, `import_export`, `search_results`), not after a file.
- **Cell layout.** Cell 1 is a markdown title plus two or three sentences: what this area is and what the reader will see. Then imports in separate cells, grouped under a comment (standard library, third-party, local). Then constants (a `SEED`, a random generator built from it, an output folder). Then one `##` section per behaviour: a short markdown cell saying what is about to happen, then small code cells, each doing one thing and ending in something visible.
- **Fixed seeds.** Every random source starts from a named seed, so two runs give the same picture.
- **Visible output.** Prefer an inline picture (matplotlib, an image shown inline), a table (rich or a dataframe), or a short printed summary. Print the values that matter, not everything.
- **Cheap assertions.** Close each section with one or two plain `assert` lines on invariants (a count, a shape, a range, "output is not empty"). They turn a notebook that only shows into one that also fails.
- **Headless.** No window pops up. The notebook must run on a server with no display. If a feature needs hardware or a service (a GPU, a display, a network), declare it in the notebook's metadata (for example `"requires": ["display"]`) and say so in the first cell. The runner then prints SKIP with the reason when it is missing.
- **Slow notebooks.** A notebook that takes more than about a minute is marked in its metadata (`"slow": true`). The default runs leave it out; an explicit flag includes it.
- **Writes only to a scratch folder** such as `__data__/` (git-ignored). Never into the source tree, never into the user's real data.
- **Outputs are committed.** Save the notebook with its outputs so it reads as a tutorial on the code host without running anything. Keep them small: downscale pictures, avoid hundreds of KB of embedded image data. A notebook that was skipped on the machine that saved it carries no outputs until someone runs it where it works.
- **Formatting.** Format the code cells with the project's formatter if it supports notebooks. Keep cell source free of paths that exist on only one machine.

## What to build

1. **A runner.** One command that executes notebooks from a clean kernel in a temporary folder, with a timeout per notebook (an option), and prints one line each: name, PASS, FAIL or SKIP, seconds. With file arguments it runs only those; with none it runs all except the slow ones; a flag includes the slow ones. A failing cell prints the cell number and the error. The exit code is non-zero if any notebook fails; SKIP is not a failure. Use the lightest tool that does this (for example `nbclient` or `jupyter nbconvert --execute`). Do not add a dependency without telling me its name, source and size first. Put the command in the project's README or contributor notes.
2. **A coverage table.** A short markdown file next to the notebooks (`tests/functional/README.md`) with one row per public package or feature area: the notebook that covers it, or "none yet". The rule to hold afterwards: every public feature area has a row, and adding a feature area means adding a row. Add a small unit test that fails when an area has no row.
3. **The notebooks.** Start with the feature areas a newcomer meets first, in the conventions above.
4. **Hooks, not CI.** Wire the runner into the project's pre-commit setup with two hooks:
   - At commit: only the notebooks in the commit (match `^tests/functional/.*\.ipynb$`), with a short timeout.
   - At push: all notebooks, without file arguments, `pass_filenames: false`, `always_run: true`, `stages: [pre-push]`. This catches a library change that breaks a notebook nobody edited.

   Pre-commit installs the pre-push hook only with `pre-commit install --hook-type pre-push`, so document that command. Say in the README that skipping hooks with `--no-verify` skips this check. Propose a CI job only if I ask for one.

## How to work

- First read the project: its README, contributor notes, package layout, how tests and the formatter are run, whether Jupyter tooling is already a dependency, and whether a pre-commit config exists. Then ask me questions (at most four at a time, recommended option first): which feature areas get a notebook and in what order, which runner you propose and what it costs to add, what should happen where a notebook needs hardware or a service that is not always there, and which notebooks count as slow.
- Do the work one notebook per commit. Run each notebook before you commit it, and tell me what you ran and what you could not run.
- Match the project's own style for code, comments and docs. Where it has no rule, keep notebooks plain: short cells, no clever helpers, no hidden state between sections that a reader cannot see.
- Do not change production code to make a notebook pass. If a notebook exposes a bug, record it as a finding and ask me.
