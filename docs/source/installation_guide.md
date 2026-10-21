# Installation Guide

This document describes how to install **ARIEL**, a research‑oriented evolutionary robotics and artificial life framework. The guide prioritizes **reproducibility, dependency stability, and ease of use** for both research and teaching environments.

We strongly recommend installing with **uv**. While a pip‑based installation is technically possible, it is discouraged due to frequent dependency conflicts.

---

## Supported Platforms

* Windows 10/11
* MacOS
* Linux

---

## Prerequisites

### Required for all installation methods

* Python 3.12+ _(mandatory)_
* Git _(optional)_
  * If you do not wish to use git, you can download the library from the GitHub page as a zip.
### For installation with `uv`

* Follow the installation instructions from the [official uv page](https://docs.astral.sh/uv/getting-started/installation/)

---

## Installation Using `uv`

> This method installs ARIEL directly from the GitHub repository.

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/ci-group/ariel.git
# Enter the cloned folder
cd ariel
```

2. **Create a virtual environment**

```bash
uv venv
```

3. **Install and synchronize dependencies**

The `uv sync` command, installs and/or syncs all dependencies needed for the library to function
```bash
uv sync
```

4. **Run an example**

```bash
uv run examples/re_book/1_brain_evolution.py
```

`uv run` automatically executes the script inside the managed virtual environment; manual activation is not required.

5. **Extending Usability**

If you want to add a dependency to the environment, you can do so using two methods.

1. Add dependency to the environment
This option adds the dependency to the uv.lock file. This means that it will remain part of your environment until you uninstall it.
```bash
uv add <dependency>
```

2. Install dependency, but don't add it to the environment
This option installs the dependency in your current environment, but does not add it to the uv.lock file. This means that after reloading (re-activating) the environment, the dependency will not be installed.
```bash
uv pip install <dependency>
```

---

## Verifying the Installation

To verify that ARIEL is installed correctly, run one of the included examples.

```bash
uv run examples/a_mujoco/1_render_single_frame.py
```

A successful run should execute without import errors and produce either rendered output or logged evolutionary progress, depending on the example.

The output of `1_render_single_frame.py` should look something like this:

![Example](../resources/example_image.jpg)

---

## Troubleshooting

* **Python version errors**: Ensure Python 3.12+ is used. Earlier versions are unsupported.
* **Dependency conflicts (uv)**: Run `uv sync` again or remove the `.venv` directory and recreate the environment.
* **Unexpected runtime errors**: Try a clean environment: remove the `.venv` directory and run `uv sync` again.

For persistent issues, consult the project documentation or repository issue tracker.

---

## Citation

If you use ARIEL in academic work, please cite the corresponding project publication (citation information will be added soon).
