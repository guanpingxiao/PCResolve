# PCResolve Output Contract

PCResolve 1.0.4 introduced the first stable provenance JSON contract.
PCResolve 1.0.5 freezes the lexical-scope-only interface documented here.
JSON outputs before 1.0.4 are experimental and not guaranteed compatible.

## CLI

```bash
pcresolve project                  # human summary
pcresolve project --json           # full provenance JSON
pcresolve project --json-summary   # compact summary JSON (CI)
pcresolve project --explain-library numpy
pcresolve project --explain-symbol x
pcresolve project --explain-call "np.array"
pcresolve project --strict
pcresolve project --usage-summary --top 20
printf '/path/to/project\n' | pcresolve --stdin --json-summary
```

- Lexical scope analysis is the only supported scope semantics.
- `--json` is the primary machine-consumption format.
- `--json-full` and `--json-stable` are hidden aliases for `--json`.
- `--json-summary` is the recommended CI format.
- `--strict` exits non-zero when error diagnostics are present.
- `--verbose`, `--usage-summary`, `--quiet`, and `--top` control text output.
- `--stdin` reads the project root from standard input.

## Python API

The stable entry points use these signatures:

```python
analyze_project(project_root)
analyze_source(source, file_path="<string>")
ProjectAnalyzer(project_root)
SingleFileAnalyzer(module_name=None, is_package=False, file_path="")
```

PCResolve uses one lexical scope model. The removed `scope_model` selector and
the former `stats.scope_model` field are not part of the 1.0.5 interface.

## Full provenance JSON (`--json`)

`all_api_calls` is the primary consumption surface. `all_symbol_provenance`
provides the explanation and evidence layer: it records which imports,
aliases, assignments, parameters, returns, attributes, containers, and
decorators support each call classification.

```json
{
  "schema_version": "1.0",
  "profile": "full",
  "project_root": ".",
  "stats": {},
  "diagnostics": [],
  "files": [],
  "all_api_calls": [],
  "all_symbol_provenance": [],
  "library_usage": {}
}
```

### `all_api_calls[*]` stable fields

| Field | Type | Description |
|-------|------|-------------|
| `expression` | string | Full call expression text |
| `func_name` | string | Function name without arguments |
| `parameters` | string | Argument text |
| `top_library` | string | Resolved primary owner: import-backed top-level library name, or `local`, `python`, `unknown`. PCResolve does not distinguish stdlib from PyPI: any import-backed owner keeps its top-level name (e.g. `json`, `pathlib`, `requests`, `numpy`). |
| `base_symbol` | string | Root/base symbol used for resolution |
| `reason` | string | DIRECT_IMPORT, RETURN_PROPAGATION, FLOW_MERGE, ... |
| `confidence` | float | 0.0–1.0 |
| `alternatives` | list | Alternative top libraries |
| `decorated_by` | list | Decorator library evidence |
| `file_path` | string | Relative POSIX path from project_root |
| `lineno` | int | |
| `col_offset` | int | |
| `end_lineno` | int | |
| `end_col_offset` | int | |
| `chain` | list | Trace chain |
| `resolved_func` | string | Fully qualified function path |
| `resolved_chain` | list | Resolved trace chain |

### `resolved_func` semantics

`resolved_func` attempts to qualify the called function through
its receiver's provenance. It is a best-effort display hint, not a
guaranteed precise resolution:

- **Constructor calls**: when PCResolve identifies the receiver class
  via `import_from_symbols`, `resolved_func` includes the class path
  (e.g. `requests.Session.get`, `flask.Flask.test_client`).
- **Factory returns**: when the receiver traces through a local
  function that returns an import-backed library-owned object, the class info is
  typically not preserved.  PCResolve may produce a library-level
  function (e.g. `requests.get`) when the call path normalizes;
  otherwise the original receiver expression is preserved while
  `top_library` carries the ownership.
- **Local / unknown**: stays as the original expression or `local`.
- **Zero-argument `super()` methods**: for a statically known direct single
  external base, the hint uses the base's import path and method name
  (e.g. `tensorflow.keras.layers.Layer.get_config`). Unsupported or ambiguous
  inheritance and explicit `super(...)` retain the original function name.
  This does not identify the method's internal defining class.
  Class decorators remain conservative, with a narrow exception for an
  import-backed `tensorflow.keras.utils.register_keras_serializable(...)`
  call using literal `package`/`name` configuration on a module-level class.
  This requires a direct named import (including a symbol alias); wildcard
  imports, visible rebinding, and module-attribute decorator calls remain
  unsupported.
  Its [registration implementation](https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/python/keras/utils/generic_utils.py)
  returns the original class; unknown or ambiguous decorators still prevent
  expansion, including when stacked with this registration decorator.
- `resolved_func` must not be treated as an importable symbol; it
  may not exist at that path in the library.

`resolved_chain` is `[func_name, resolved_func, top_library]`.

## Summary JSON (`--json-summary`)

```json
{
  "schema_version": "1.0",
  "profile": "summary",
  "project_root": ".",
  "stats": {},
  "diagnostics": [],
  "libraries": {}
}
```

Summary excludes `all_api_calls`, `all_symbol_provenance`,
and per-file `symbols`/`chains`.

## Path normalization

All paths use POSIX separators (`/`) relative to `project_root`.
External paths use the `<external>/...` prefix.

## Reason constants

| Reason | Meaning |
|--------|---------|
| DIRECT_IMPORT | Call traced directly to an import alias or from-import |
| TRANSITIVE_IMPORT | Call traced through a re-export or transitive module chain |
| LOCAL_DEFINITION | Call is a locally defined function, method, or class |
| BUILTIN | Call is a Python builtin (no import required) |
| PARAMETER_PROPAGATION | Source traced through a function parameter |
| RETURN_PROPAGATION | Source traced through a function return value |
| FLOW_MERGE | Multiple branches/sources merged (if/else, multi-return, SourceSet) |
| UNRESOLVED | Trace could not reach a terminal origin |

`PARAMETER_PROPAGATION` remains a stable reason value. In 1.0.5, parameter
evidence often resolves before final classification, so the resulting call may
instead report `RETURN_PROPAGATION`, `FLOW_MERGE`, or `TRANSITIVE_IMPORT`.

## Confidence rules

| Reason | Confidence |
|--------|-----------|
| DIRECT_IMPORT | 1.0 |
| LOCAL_DEFINITION | 1.0 |
| BUILTIN | 1.0 |
| PARAMETER_PROPAGATION | 0.9 |
| RETURN_PROPAGATION | 0.9 |
| TRANSITIVE_IMPORT | 0.9 |
| FLOW_MERGE (single) | 0.85 |
| FLOW_MERGE (N alts) | max(1/N, 0.2) |
| FLOW_MERGE with a conservative local primary | 0.5 |
| UNRESOLVED | 0.0 |

## Version History

- **1.0.5:** lexical scope analysis is the only scope semantics; JSON `stats`
  contains parsed, skipped, and total module counts only.
- **1.0.4:** `--json` became the full provenance schema;
  `--json-stable` became a deprecated hidden alias.
- **Before 1.0.4:** JSON was experimental and has no compatibility guarantee.
