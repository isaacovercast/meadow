# Plan: Rename the Package from `meadow` to `meadow`

## 1. Goal
Rename the Python package and import namespace from:
- `meadow`

to:
- `meadow`

The package should remain a normal `src/`-layout Python package.

That means the target layout is:
- `src/meadow/__init__.py`
- `src/meadow/graph.py`
- `src/meadow/train.py`
- etc.

This plan explicitly does **not** flatten modules directly into `src/`, because that would make the packaging and imports worse.

No implementation is performed in this plan.

## 2. Desired End State
After the rename:
- source code lives in `src/meadow/`
- all internal imports use `meadow`
- all public examples/docs/notebooks import from `meadow`
- package metadata refers to the new project/package identity
- no remaining imports or references to `meadow` remain except historical prose if intentionally preserved
 - Do not retain any references to `meadow` as there is no historical need

## 3. High-Level Strategy
This should be done as a namespace-preserving package rename, not as a packaging redesign.

That means:
1. rename the package directory
2. update all imports
3. update packaging metadata
4. update docs/examples/notebooks
5. regenerate or update egg-info metadata
6. run compile/import validation

## 4. Directory/Layout Changes

## 4.1 Rename the package directory
Current:
- `src/meadow/`

Target:
- `src/meadow/`

All files currently inside `src/meadow/` should move into `src/meadow/` unchanged unless import edits are required.

Expected contents after move:
- `src/meadow/__init__.py`
- `src/meadow/climate.py`
- `src/meadow/cv.py`
- `src/meadow/data.py`
- `src/meadow/graph.py`
- `src/meadow/io.py`
- `src/meadow/model.py`
- `src/meadow/raster.py`
- `src/meadow/train.py`
- `src/meadow/utils.py`
- `src/meadow/vcf_to_hdf5.py`
- `src/meadow/viz.py`

## 4.2 Remove the old package directory
After the move, `src/meadow/` should no longer exist.

No compatibility shim package should be left behind unless explicitly requested later.

## 5. Import Refactor
This is the main code-level change.

## 5.1 Internal absolute imports
All imports like:
```python
from meadow.graph import ...
from meadow.data import ...
```

must become:
```python
from meadow.graph import ...
from meadow.data import ...
```

This affects the source files under the renamed package.

## 5.2 Search scope for import updates
Search and replace across:
- `src/meadow/*.py`
- `examples/`
- `notebooks/`
- `README.md`
- `docs/`
- any generated package metadata that is tracked

## 5.3 Avoid mixed namespace residue
There should be no surviving runtime imports from `meadow`.

Any mixed state where some modules import from `meadow` and some from `meadow` should be treated as a failed partial migration.

## 6. Packaging Metadata Changes

## 6.1 `pyproject.toml`
File:
- `pyproject.toml`

Required changes:
1. change `[project].name`
2. ensure package discovery still works under `src/`

### Naming recommendation
Use:
- distribution name: `meadow`

unless you intentionally want a different install name from the import name.

Current likely fields to update:
- `name = "meadow"`
- description text if it uses the old package identity
- keywords if desired

The `package-dir = {"" = "src"}` and package discovery under `src` can stay as-is.

## 6.2 Egg-info metadata
Current tracked metadata lives under:
- `src/meadow.egg-info/`

This needs special handling.

### Recommended approach
Because it is tracked, update or regenerate it so it matches the new package name.

Likely changes:
- rename egg-info directory if appropriate
- update:
  - `PKG-INFO`
  - `SOURCES.txt`
  - `top_level.txt`
  - other generated metadata files if tracked

### Important note
If egg-info is meant to remain generated-only in the future, that is a separate cleanup decision. For this rename, the key requirement is simply that tracked metadata must not keep stale references to `meadow`.

## 7. Public API Surface

## 7.1 `__init__.py`
File:
- `src/meadow/__init__.py`

This file should remain the public package surface, but with the new package path.

The export list itself does not need conceptual redesign; only the package path changes.

## 7.2 User-facing imports
All user-facing docs/examples should move from:
```python
from meadow.train import build_species_graphs
```

to:
```python
from meadow.train import build_species_graphs
```

## 8. Documentation and Example Updates

## 8.1 README
File:
- `README.md`

Required updates:
- package name references
- code snippets
- install/import examples
- any prose describing `meadow`

## 8.2 MkDocs pages
Files under:
- `docs/`

Required updates:
- code snippets
- module references
- package-qualified file/module names in prose
- architecture or overview text referencing the old namespace

## 8.3 Notebooks
Files under:
- `notebooks/`

Required updates:
- import cells
- any explanatory text mentioning `meadow`

Notebook JSON validity must be rechecked after edits.

## 8.4 Examples
Files under:
- `examples/`

Required updates:
- `sys.path` usage if needed
- imports from `meadow` to `meadow`

## 9. Files and Strings to Search For
At minimum, search the repo for these strings and update them appropriately:
- `meadow`
- `meadow`
- `src/meadow`
- `meadow.egg-info`

Not every occurrence of `meadow` in prose must necessarily become `meadow`, but every packaging/import identity should.
 - Flag any instances of `meadow` that are not changed in this process, by this i mean report them after the fact.

## 10. Non-Goals
This rename should **not**:
- flatten modules directly into `src/`
- redesign the package architecture
- change function signatures unless needed for import paths
- add a compatibility alias package unless explicitly requested

## 11. Implementation Sequence
1. Rename `src/meadow/` to `src/meadow/`.
2. Update all source-file imports under `src/meadow/`.
3. Update `pyproject.toml` project/package naming.
4. Update tracked egg-info metadata under `src/*.egg-info`.
5. Update examples, README, docs, and notebooks.
6. Search for stale references to the old namespace.
7. Run compile validation.
8. Re-parse notebooks as JSON.

## 12. Validation Plan
Minimum validation after implementation:
- `python -m py_compile src/meadow/*.py examples/minimal_prototype.py`
- notebook JSON parse check for all notebooks
- repo-wide search confirming no stale import references to `meadow`

If the environment supports it, also do:
- one import smoke test such as:
  - `from meadow.train import build_species_graphs`
  - `from meadow.graph import SpeciesGraph`

## 13. Expected Result
After the rename, the project will have:
- package/import name: `meadow`
- standard `src/meadow/` layout
- consistent internal and public imports
- updated docs/examples/notebooks
- no residual dependency on the old package namespace
