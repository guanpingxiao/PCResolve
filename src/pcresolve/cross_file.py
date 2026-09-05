## @package pcresolve.cross_file
#  Provide cross-file project-level API call chain analysis.
#
#  The ProjectAnalyzer class orchestrates scanning a project, parsing every
#  .py file, building per-file symbol tables, resolving symbols across files,
#  and collecting all API calls with their top-level library origins.

import ast
import os
import builtins
from .module_mapper import ModuleMapper

## Python 2 builtins not present in Python 3's builtins module.
_PY2_BUILTINS = frozenset({
    "apply", "basestring", "buffer", "cmp", "coerce", "execfile",
    "file", "intern", "long", "raw_input", "reduce", "reload",
    "StandardError", "unichr", "unicode", "xrange",
})

## 1.0.5 P1: builtin container/type methods keyed by container kind.
_BUILTIN_CONTAINER_METHODS = {
    "list": frozenset([
        "append", "extend", "insert", "remove", "pop", "clear",
        "index", "count", "sort", "reverse", "copy", "__len__",
    ]),
    "dict": frozenset([
        "get", "keys", "values", "items", "update", "pop",
        "popitem", "clear", "copy", "__len__",
    ]),
    "set": frozenset([
        "add", "remove", "discard", "pop", "clear", "copy",
        "update", "difference", "intersection", "union",
        "symmetric_difference", "issubset", "issuperset", "__len__",
    ]),
    "tuple": frozenset(["count", "index", "__len__"]),
    "str": frozenset([
        "strip", "rstrip", "lstrip", "split", "rsplit", "join",
        "replace", "find", "rfind", "rindex", "startswith",
        "endswith", "upper", "lower", "title", "capitalize",
        "swapcase", "center", "ljust", "rjust", "encode", "zfill",
        "format", "format_map",
        "isalnum", "isalpha", "isascii", "isdecimal", "isdigit",
        "isidentifier", "islower", "isnumeric", "isprintable",
        "isspace", "istitle", "isupper", "__len__",
    ]),
}

## Check if a receiver name has a known container kind from single-file analysis.
#  @param tracer Single-file analyzer.
#  @param receiver_name Variable name.
#  @return Container kind string or None.
def _receiver_container_kind(tracer, receiver_name):
    return getattr(tracer, "container_kinds", {}).get(receiver_name)


## Check if a name is a Python builtin(including Python 2 builtins).
def _is_builtin(name):
    return isinstance(name, str) and (hasattr(builtins, name) or name in _PY2_BUILTINS)


## Select one field from a bounded tuple/list source.
#  @param source Candidate tuple/list source.
#  @param index Zero-based field index.
#  @return Field source, or None when the field is not proven.
def _tuple_source_item(source, index):
    source = normalize_source(source)
    if not isinstance(source, TupleSource) or not isinstance(index, int):
        return None
    if index < 0 or index >= len(source.items):
        return None
    return normalize_source(source.items[index])
from .diagnostics import Diagnostic, FILE_READ_ERROR, SYNTAX_ERROR, ENCODING_ERROR
from .ir import (SymbolProvenance, ClassificationResult,
                    REASON_DIRECT_IMPORT)
from .single_file import (SingleFileAnalyzer, _has_result_owner_contract,
                          _match_result_owner, _is_verified_result_owner,
                          _has_builtin_shape_method,
                          _builtin_method_return_shape,
                          _match_result_python_shape)
from .sources import (ContainerItem, ContainerIter, TupleSource, InstanceMethod,
                       ParameterSource, InstanceAttribute, PythonShape,
                       SuperMethod, CallResult,
                       DerivedResult, UnknownSource,
                       SourceSet, is_structured_source, normalize_source,
                       source_display, make_source_set)
from .call_graph import CallContext, FunctionId, ProjectCallGraph
from .classification import classify_confidence, ClassificationPipeline
from .decorator_provenance import build_decorator_index, lookup_decorated_by
from .library_usage import build_library_usage
from .source_resolution import SourceSetResolver
from .types import ProjectAnalysis, FileAnalysis, ApiCall


## Remove consecutive duplicate items from a list while preserving order.
#  @param chain Input list.
#  @return List with no consecutive duplicates.
def _dedup_consecutive(chain):
    result = []
    for item in chain:
        if not result or item != result[-1]:
            result.append(item)
    return result


## Check whether a symbol is an imported external origin in this tracer.
#  @param tracer Single-file analyzer.
#  @param symbol Candidate external origin.
#  @return True if symbol matches an import source or its top-level package.
def _is_import_origin(tracer, symbol):
    if tracer is None or not isinstance(symbol, str):
        return False
    def _matches(value):
        if not isinstance(value, str):
            return False
        return (value == symbol
                or value.startswith(symbol + ".")
                or symbol.startswith(value + "."))
    for value in tracer.symbols.direct.values():
        if isinstance(value, SourceSet):
            for src in value.sources:
                if _matches(src):
                    return True
        elif _matches(value):
            return True
    # Function-local imports live in lexical scopes instead of the module-level
    # compatibility table. SymbolRef is the durable project
    # fact available after the visitor has left that scope.
    for ref in getattr(tracer, "symbol_refs", []):
        if ref.kind == "import" and _matches(normalize_source(ref.source)):
            return True
    return False


## Check whether a source contains explicit project return evidence.
#  This distinguishes a receiver propagated through a local return or tuple
#  binding from an independently verified external result contract.  The
#  latter may classify the current call, but it must not rewrite unrelated
#  downstream records as new call-graph evidence.
#  @param source Source value to inspect.
#  @param _seen Recursion guard for nested source objects.
#  @return True when the source contains a return-origin SourceSet.
def _has_return_provenance(source, _seen=None):
    source = normalize_source(source)
    if source is None:
        return False
    if _seen is None:
        _seen = set()
    marker = id(source)
    if marker in _seen:
        return False
    _seen.add(marker)
    if isinstance(source, SourceSet):
        if source.origin == "return":
            return True
        return any(_has_return_provenance(item, _seen)
                   for item in source.sources)
    if isinstance(source, CallResult):
        return _has_return_provenance(source.result_source, _seen)
    if isinstance(source, DerivedResult):
        return any(_has_return_provenance(item, _seen)
                   for item in source.sources)
    if isinstance(source, InstanceMethod):
        return _has_return_provenance(source.receiver, _seen)
    return False


## Build an ApiCall from a get_calls() record dict.
#  @param c Call record dict.
#  @param deco_by Decorator evidence index.
#  @return ApiCall object.
def _make_api_call(c, deco_by):
    """Build an ApiCall from a get_calls() record dict."""
    return ApiCall(
        expression=c['api'],
        top_library=c['top'],
        base_symbol=source_display(c.get('base', '')),
        chain=c.get('chain', []),
        file_path=c.get('file_path', ''),
        lineno=c.get('lineno', 0),
        col_offset=c.get('col_offset', 0),
        end_lineno=c.get('end_lineno', 0),
        end_col_offset=c.get('end_col_offset', 0),
        func_name=c.get('func_name', ''),
        parameters=c.get('parameters', ''),
        resolved_func=c.get('resolved_func', ''),
        resolved_chain=[c.get('func_name', ''), c.get('resolved_func', ''), c.get('top', '')],
        reason=c.get('reason', ''),
        confidence=c.get('confidence', 1.0),
        alternatives=c.get('alternatives', []),
        decorated_by=lookup_decorated_by(
            c.get('file_path', ''),
            c.get('func_name', ''),
            c.get('scope_name', ''), deco_by),
    )


## Cross-file project analyzer that traces all API calls to their origins.
#
#  Steps:
#  1. Scan the project for all .py/.pyi files and map them to module names.
#  2. Parse each file and run SingleFileAnalyzer to build per-file symbol data.
#  3. Resolve cross-file symbol references across the project.
#  4. Classify every API call with its top-level library source.
class ProjectAnalyzer:
    # ── pipeline ───────────────────────────────────────────────────────

    ## Initialize the analyzer for a given project root.
    #  @param project_root Absolute path to the project root directory.
    def __init__(self, project_root):
        self.project_root = project_root
        self.module_mapper = ModuleMapper(project_root)
        self.global_symbols = {}
        self.symbol_chains = {}
        self.all_calls = {}
        self._python_shape_in_progress = set()
        self._callable_field_in_progress = set()
        self._constructor_only_fields = None
        self._source_resolver = SourceSetResolver(
            top_source_cb=self._top_source,
            cg_return_cb=self._lookup_cg_return_source,
            known_local_cb=self._is_known_local_symbol,
            resolve_structured_cb=self._resolve_structured_source,
            dedupe_cb=self._dedupe_list,
            is_import_origin_cb=_is_import_origin)
        self._pipeline = ClassificationPipeline(
            origin_candidates_cb=self._origin_candidates,
            is_direct_import_cb=self._is_direct_import_base,
            dedupe_cb=self._dedupe_list)

    ## Run the full analysis: scan, parse, resolve, and collect.
    #  @return ProjectAnalysis with all results.
    def analyze(self):
        self.module_mapper.scan_project()
        all_modules = self.module_mapper.get_all_modules()
        module_tracers = {}
        diagnostics = []

        for module in all_modules:
            file_path = self.module_mapper.get_file_path(module)
            if not file_path or not os.path.exists(file_path):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except UnicodeDecodeError as e:
                diagnostics.append(Diagnostic(
                    code=ENCODING_ERROR,
                    message="Cannot decode file: %s" % e,
                    severity="error",
                    file_path=file_path,
                    module_name=module,
                ))
                continue
            except OSError as e:
                diagnostics.append(Diagnostic(
                    code=FILE_READ_ERROR,
                    message="Cannot read file: %s" % e,
                    severity="error",
                    file_path=file_path,
                    module_name=module,
                ))
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                diagnostics.append(Diagnostic(
                    code=SYNTAX_ERROR,
                    message=str(e),
                    severity="error",
                    file_path=file_path,
                    lineno=getattr(e, 'lineno', 0),
                    col_offset=getattr(e, 'offset', 0) if getattr(e, 'offset', 0) else 0,
                    end_lineno=getattr(e, 'end_lineno', 0) or 0,
                    end_col_offset=getattr(e, 'end_offset', 0) if getattr(e, 'end_offset', 0) else 0,
                    module_name=module,
                ))
                continue
            tracer = SingleFileAnalyzer(
                module_name=module,
                is_package=self.module_mapper.is_package(module),
                file_path=file_path,
            )
            tracer.visit(tree)
            module_tracers[module] = tracer

        ## Aggregate per-module call-graph facts (Phase 7B-full PR1).
        self.project_cg = ProjectCallGraph()
        for module, tracer in module_tracers.items():
            if (tracer.module_cg.functions or tracer.module_cg.classes
                    or tracer.module_cg.edges
                    or tracer.module_cg.iteration_bindings):
                self.project_cg.modules[module] = tracer.module_cg

        self._bind_bounded_local_call_results(module_tracers)
        self._bind_bounded_callback_map_results(module_tracers)
        self._bind_bounded_local_iteration_results(module_tracers)
        self._bind_proven_result_method_results(module_tracers)
        self.resolve_cross_file_symbols(module_tracers)
        self.get_calls(module_tracers)

        all_provenance = self._build_symbol_provenance(module_tracers)
        deco_by = self._build_decorator_index(all_provenance)

        files = self._build_file_analysis(module_tracers, all_provenance, deco_by)
        all_api_calls = self._build_all_api_calls(deco_by)
        library_usage = self._build_library_usage(all_api_calls, all_provenance)

        stats = {
            "total_modules": len(all_modules),
            "parsed_modules": len(module_tracers),
            "skipped_modules": len(diagnostics),
        }

        return ProjectAnalysis(
            project_root=self.project_root,
            files=files,
            all_api_calls=all_api_calls,
            diagnostics=diagnostics,
            stats=stats,
            all_symbol_provenance=all_provenance,
            library_usage=library_usage,
        )

    # ── provenance helpers ───────────────────────────────────────────────

    ## Check whether a top name is backed by any import evidence across tracers.
    def _is_prov_import_backed(self, name, tracers):
        if not isinstance(name, str) or '.' in name:
            return bool('.' in name) if isinstance(name, str) else False
        for tr in tracers.values():
            if _is_import_origin(tr, name):
                return True
            if name in getattr(tr, 'import_aliases', set()):
                return True
        return False

    # ── output construction ──────────────────────────────────────────────

    ## Build per-file analysis results.
    #  @param module_tracers Dict of module_name -> SingleFileAnalyzer.
    #  @param all_provenance List of SymbolProvenance records.
    #  @param deco_by Decorator evidence index.
    #  @return List of FileAnalysis records.
    def _build_file_analysis(self, module_tracers, all_provenance, deco_by):
        files = []
        for module, tracer in module_tracers.items():
            file_path = self.module_mapper.get_file_path(module)
            files.append(FileAnalysis(
                file_path=file_path,
                module_name=module,
                symbols=self.global_symbols.get(module, {}),
                chains=self.symbol_chains.get(module, {}),
                symbol_provenance=[p for p in all_provenance
                                   if p.file_path == file_path],
                api_calls=[_make_api_call(c, deco_by)
                           for c in self.all_calls.get(module, [])],
            ))
        return files

    ## Build the flat project-level API call list.
    #  @param deco_by Decorator evidence index.
    #  @return List of ApiCall records.
    def _build_all_api_calls(self, deco_by):
        return [_make_api_call(c, deco_by)
                for module, calls in self.all_calls.items()
                for c in calls]

    ## Build a decorator evidence index from provenance records.
    #  @param all_provenance List of SymbolProvenance records.
    #  @return Dict keyed by (file_path, scope, symbol) → [library, ...].
    def _build_decorator_index(self, all_provenance):
        return build_decorator_index(all_provenance)

    ## Build SymbolProvenance records from each tracer's symbol_refs.
    #  @param module_tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return List of SymbolProvenance records.
    def _build_symbol_provenance(self, module_tracers):
        result = []
        for module, tracer in module_tracers.items():
            file_path = self.module_mapper.get_file_path(module)
            for ref in tracer.symbol_refs:
                direct_source = ref.source
                if ref.kind == "parameter" and ref.scope_name:
                    direct_source = ParameterSource(
                        ref.scope_name, ref.symbol)
                try:
                    chain = self.trace_symbol(module, ref.symbol, module_tracers,
                                               set(),
                                               _direct_source=direct_source)
                except RecursionError:
                    chain = [source_display(ref.source)]
                chain = _dedup_consecutive(chain)
                top = self.extract_final_source(chain) if chain else ""
                if top and top not in ("local", "python", "unknown", ""):
                    if '.' not in top and not self._is_prov_import_backed(top, module_tracers):
                        ## 1.0.5 P2: explicit result_source confirms library
                        #  identity even without import statement evidence.
                        rs = getattr(ref.source, 'result_source', None)
                        if (isinstance(ref.source, CallResult)
                                and isinstance(rs, str)
                                and rs not in ("local", "python", "unknown", "")
                                and rs.split(".")[0] == top):
                            pass
                        else:
                            top = "local"
                tops = [top] if top else []
                cr = self.classify_source(
                    ref.source, top, module, tracer, module_tracers,
                    expand_origins=False, symbol=ref.symbol, kind=ref.kind)
                prov = SymbolProvenance(
                    symbol=ref.symbol,
                    kind=ref.kind,
                    top_libraries=tops,
                    top_library=tops[0] if tops else "unknown",
                    chain=chain,
                    scope_name=ref.scope_name,
                    file_path=file_path or "",
                    lineno=ref.lineno,
                    col_offset=ref.col_offset,
                    reason=cr.reason,
                    confidence=cr.confidence,
                    alternatives=cr.alternatives,
                )
                result.append(prov)
        return result

    # ── library usage ────────────────────────────────────────────────────

    ## Build a library usage index from calls and provenance.
    #
    #  Delegates to library_usage.build_library_usage() (Phase 9-lite PR1).
    #  @param all_api_calls List of ApiCall records.
    #  @param all_provenance List of SymbolProvenance records.
    #  @return Dict of library_name -> LibraryUsage.
    def _build_library_usage(self, all_api_calls, all_provenance):
        return build_library_usage(
            self.project_root, all_api_calls, all_provenance)

    ## Check whether a module name belongs to the current project.
    #  @param module_name Dotted module name.
    #  @return True if the module is a local project module.
    def is_local(self, module_name):
        return module_name in self.module_mapper.get_all_modules()

    ## Downgrade an import-backed method owner after a visible monkey patch.
    #
    #  A patch on one imported class does not prove that every receiver from
    #  that library has the patched class. When the receiver has already been
    #  reduced to the library owner, the only sound primary is unknown.
    #  @param call_detail Single-file call record.
    #  @param tracer Analyzer for the current module.
    #  @param top_source Resolved import-backed owner.
    #  @return top_source or "unknown".
    def _apply_external_override_ambiguity(
            self, call_detail, tracer, top_source):
        if top_source in ("", None, "local", "python", "unknown"):
            return top_source
        func_name = call_detail.get("func_name", "")
        if "." not in func_name:
            return top_source
        method_name = func_name.rsplit(".", 1)[-1]
        patches = getattr(
            tracer, "external_method_overrides", {}).get(
                (top_source, method_name), [])
        if not patches:
            return top_source
        call_scope = call_detail.get("scope_name", "")
        call_line = call_detail.get("lineno", 0)
        for patch_scope, patch_line, _ in patches:
            visible = (
                patch_scope == ""
                or (
                    patch_scope == call_scope
                    and patch_line <= call_line
                )
            )
            if visible:
                return "unknown"
        return top_source

    ## Promote a chained local-call receiver to structured result evidence.
    #
    #  Single-file collection intentionally keeps ``make_value().method``
    #  conservative because the return object is resolved only after the
    #  project call graph is available.  Once the exact local edge is known,
    #  represent the receiver as a CallResult so the existing return-summary
    #  resolver can follow it.  This is limited to an unambiguous project
    #  call edge and never infers an external library from the method name.
    #  @param module Current caller module.
    #  @param call_detail Raw single-file call record.
    #  @param module_tracers All module analyzers.
    #  @return InstanceMethod receiver, or None when the edge is unresolved.
    def _promote_chained_local_call_receiver(
            self, module, call_detail, module_tracers):
        base = normalize_source(call_detail.get("base"))
        func_name = call_detail.get("func_name", "")
        if not isinstance(base, str) or not isinstance(func_name, str):
            return None
        marker = "()."
        if func_name.startswith(base + marker):
            inner_name, _, suffix = func_name.partition(marker)
        else:
            # The legacy base may be a resolved class name while func_name
            # retains the source receiver spelling.  Recover the inner call
            # from the recorded expression instead of requiring those two
            # representations to share a prefix.
            try:
                expression = ast.parse(
                    call_detail.get("api", ""), mode="eval").body
            except (SyntaxError, ValueError, TypeError):
                return None
            if (not isinstance(expression, ast.Call)
                    or not isinstance(expression.func, ast.Attribute)
                    or not isinstance(expression.func.value, ast.Call)):
                return None
            inner_name = ast.unparse(expression.func.value.func)
            suffix = expression.func.attr
        if not inner_name or not suffix:
            return None
        edges = [
            edge for edge in self.project_cg.modules.get(
                module, ProjectCallGraph()).edges
            if edge.caller.qualname == (
                call_detail.get("scope_name", "") or "<module>")
            and edge.call_lineno == call_detail.get("lineno", 0)
            and edge.call_col_offset == call_detail.get("col_offset", 0)
            and edge.callee_name == inner_name
        ]
        if len(edges) != 1:
            return None
        targets = self._local_edge_targets(edges[0], module, module_tracers)
        if len(targets) != 1:
            return None
        target = targets[0]
        result = CallResult(
            target.module + "." + target.qualname,
            display_name=inner_name,
            call_lineno=edges[0].call_lineno,
            call_col_offset=edges[0].call_col_offset,
            source_module=module,
        )
        return InstanceMethod(result, suffix.rsplit(".", 1)[-1])

    ## Collect all API calls across all modules and resolve their top-level origin.
    #  @param module_tracers Dict of module_name -> SingleFileAnalyzer.
    def get_calls(self, module_tracers):
        self._call_searched_global = set()
        for module, tracer in module_tracers.items():
            file_path = self.module_mapper.get_file_path(module)
            for c in tracer.api_calls:
                c['file_path'] = file_path or ''

            self.all_calls[module] = []
            for call_detail in tracer.api_calls:
                base = call_detail.get('base')
                promoted_receiver = self._promote_chained_local_call_receiver(
                    module, call_detail, module_tracers)
                if (promoted_receiver is not None
                        and not isinstance(base, UnknownSource)):
                    base = promoted_receiver
                # A comprehension variable is an element of the receiver
                # container, not the owner of that container.  When a local
                # function returns a tuple and no positional binding reaches
                # the comprehension, keep the element unresolved instead of
                # promoting the homogeneous container owner.
                base_receiver = (
                    base.receiver
                    if isinstance(base, InstanceMethod) else base)
                if (call_detail.get('scope_name') == '<comprehension>'
                        and call_detail.get('func_name', '').split('.', 1)[0]
                        in getattr(tracer, 'comprehension_targets', set())
                        and isinstance(base_receiver, CallResult)
                        and base_receiver.result_source is None
                        and self._is_unbound_tuple_call_result(
                            module, base_receiver, module_tracers)):
                    base = UnknownSource("unresolved tuple element")
                    call_detail = dict(call_detail)
                    call_detail['top'] = 'unknown'
                if (call_detail.get('top') == 'unknown'
                        or isinstance(base, UnknownSource)):
                    # Preserve unknown top — the single-file phase
                    # already determined the owner cannot be resolved.
                    record = dict(call_detail)
                    record['top'] = 'unknown'
                    cr = self.classify_source(
                        base, 'unknown', module, tracer, module_tracers)
                    record['reason'] = cr.reason
                    record['confidence'] = cr.confidence
                    record['alternatives'] = cr.alternatives
                    self.all_calls[module].append(record)
                    continue
                if call_detail.get('top') == 'local':
                    if isinstance(base, str) or is_structured_source(base):
                        top_source = self._base_top_source(module, base, tracer, module_tracers)
                        if top_source and top_source != 'local':
                            top_source = self._apply_external_override_ambiguity(
                                call_detail, tracer, top_source)
                            record = dict(call_detail)
                            record['top'] = top_source
                            cr = self.classify_source(
                                base, top_source, module, tracer, module_tracers)
                            record['reason'] = cr.reason
                            record['alternatives'] = cr.alternatives
                            record['confidence'] = cr.confidence
                            self.all_calls[module].append(record)
                            continue
                    # 1.0.5 P0: container methods on local/builtin receivers
                    # must not inherit argument provenance as top_library.
                    # Argument provenance belongs in SymbolProvenance, not
                    # ApiCall.top_library.
                    record = dict(call_detail)
                    record['top'] = 'local'
                    cr = self.classify_source(
                        base, 'local', module, tracer, module_tracers)
                    record['reason'] = cr.reason
                    record['confidence'] = cr.confidence
                    record['alternatives'] = cr.alternatives
                    self.all_calls[module].append(record)
                    continue
                # 1.0.5 P1: preserve python top from single-file phase
                # for calls with call-site recorded container kind or a
                # lexically verified, unshadowed builtin callable.
                call_kind = call_detail.get("receiver_container_kind")
                direct_builtin = (
                    call_detail.get("top") == "python"
                    and call_detail.get("direct_name_callee") == base
                    and _is_builtin(base)
                )
                if ((call_kind is not None
                     and call_detail.get("top") == "python")
                        or direct_builtin):
                    record = dict(call_detail)
                    record['top'] = "python"
                else:
                    top_source = self._base_top_source(module, base, tracer, module_tracers)
                    top_source = self._apply_external_override_ambiguity(
                        call_detail, tracer, top_source)
                    record = dict(call_detail)
                    record['top'] = top_source
                cr = self.classify_source(
                    base, record['top'], module, tracer, module_tracers)
                record['reason'] = cr.reason
                record['alternatives'] = cr.alternatives
                record['confidence'] = cr.confidence
                self.all_calls[module].append(record)

        for module, tracer in module_tracers.items():
            for c in self.all_calls.get(module, []):
                c['resolved_func'] = self._resolve_func_name(c, module, tracer)

        self._call_searched_global = None

    ## Check whether a call result used in a comprehension is an unbound
    #  project-local tuple result.
    #  @param module Caller module.
    #  @param source CallResult used as the comprehension receiver.
    #  @param tracers Per-module analyzers.
    #  @return True when the tuple item position is not statically known.
    def _is_unbound_tuple_call_result(self, module, source, tracers):
        context = self._bounded_call_context(
            module, source.call_lineno, source.call_col_offset,
            tracers)
        if context is None:
            return False
        module_cg = self.project_cg.modules.get(context.target.module)
        summary = (
            module_cg.functions.get(context.target.qualname)
            if module_cg is not None else None)
        return summary is not None and self._is_tuple_return_source(
            summary.returns)

    ## Resolve the top-level source of a base symbol, preferring call_assign_funcs.
    #  @param module The current module.
    #  @param base The base symbol string.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param module_tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return Top-level library name.
    def _base_top_source(self, module, base, tracer, module_tracers):
        if is_structured_source(base):
            if (isinstance(base, InstanceMethod)
                    and isinstance(base.receiver, str)
                    and _is_verified_result_owner(base.receiver)):
                return base.receiver
            if (isinstance(base, InstanceMethod)
                    and isinstance(base.receiver, CallResult)):
                result_owner = normalize_source(
                    base.receiver.result_source)
                if isinstance(result_owner, str) and result_owner:
                    return result_owner
                candidates = self._bounded_call_result_method_candidates(
                    module, base.receiver, base.method, module_tracers,
                    None, set(), None)
                candidate = self._bounded_candidates_top(candidates or [])
                if _is_verified_result_owner(candidate):
                    # A resolved owner is not a lexical symbol. Looking it
                    # up again can redirect it through a local star import.
                    return candidate
            structured = self._resolve_structured_source(module, base, module_tracers)
            if structured is not None:
                ## Explicit result_source carries an owner, not a symbol to
                # resolve again in module scope.  InstanceMethod receivers
                # cover assignments from function-local import chains.
                result = base
                if (isinstance(base, InstanceMethod)
                        and isinstance(base.receiver, CallResult)):
                    result = base.receiver
                if isinstance(result, CallResult):
                    rs = getattr(result, 'result_source', None)
                    if (isinstance(rs, str)
                            and rs not in ("local", "python", "unknown", "")):
                        return rs
                _, src_module, src_symbol = structured
                if src_symbol in ("local", "python", "unknown"):
                    return src_symbol
                if _is_verified_result_owner(src_symbol):
                    return src_symbol
                if (isinstance(base, InstanceMethod)
                        and base.parameter_scope
                        and isinstance(src_symbol, str)
                        and src_symbol):
                    return src_symbol
                top = self._top_source(src_module, src_symbol, module_tracers)
                # 1.0.5 P1: builtin container method on a receiver
                # whose container kind is known from tracer-final state.
                # This fallback only sees module-level legacy maps.
                # Function-local receiver kinds are captured at the call
                # site and preserved before this path is reached.
                if (top == "local"
                        and isinstance(base, InstanceMethod)
                        and base.receiver not in tracer.class_methods
                        and base.receiver in tracer.symbols.direct):
                    kind = getattr(tracer, "container_item_kinds", {}).get(base.receiver)
                    if kind is None:
                        kind = getattr(tracer, "container_kinds", {}).get(base.receiver)
                    if (kind is not None
                            and base.method in _BUILTIN_CONTAINER_METHODS.get(kind, frozenset())):
                        return "python"
                return top
            return "local"
        if isinstance(base, str) and '.' in base:
            prefix = base.split('.')[0]
            if prefix in self.global_symbols.get(module, {}):
                return self.global_symbols[module][prefix]
            return self._top_source(module, base, module_tracers)
        if isinstance(base, str):
            caf = tracer.call_assign_funcs.get(base)
            if caf:
                caf_first = caf.split('.')[0]
                top = self._top_source(module, caf_first, module_tracers)
                if top and top != 'local':
                    return top
        if base in self.global_symbols.get(module, {}):
            return self.global_symbols[module][base]
        return self._top_source(module, base, module_tracers)

    ## Check whether a symbol is a known local definition in this tracer.
    #  @param tracer Single-file analyzer.
    #  @param symbol Candidate symbol name.
    #  @return True if the symbol is a local function/method/class/param.
    def _is_known_local_symbol(self, tracer, symbol):
        if not isinstance(symbol, str):
            return False
        first = symbol.split(".")[0]
        if first in ("self", "cls"):
            return True
        if first in getattr(tracer, "local", set()):
            return True
        if first in getattr(tracer, "defined_functions", set()):
            return True
        if first in getattr(tracer, "class_methods", {}):
            return True
        for methods in getattr(tracer, "class_methods", {}).values():
            if first in methods:
                return True
        direct = normalize_source(tracer.symbols.direct.get(first))
        if direct == "local":
            return True
        return False

    ## Check whether a string base represents a direct import.
    #  @param tracer Single-file analyzer.
    #  @param base Candidate base name.
    #  @return True if base is an import alias or from-import symbol.
    def _is_direct_import_base(self, tracer, base):
        if not isinstance(base, str):
            return False
        first = base.split(".")[0]
        if first in getattr(tracer, "import_aliases", set()):
            return True
        if first in getattr(tracer, "import_from_symbols", {}):
            return True
        direct = normalize_source(tracer.symbols.direct.get(first))
        if isinstance(direct, str) and direct not in ("local", "python", "unknown"):
            return True
        return False

    ## Check whether a SymbolProvenance import is a direct external import.
    #
    #  True when the import source is a non-local module and the resolved
    #  top matches the source's top-level name.  Local re-exports
    #  (local_lib -> requests) are not direct external imports.
    #  @param base The import source value (e.g. "functools").
    #  @param top The resolved top library.
    #  @param module The module where the import occurs.
    #  @return True if this is a direct external import.
    def _is_direct_external_import(self, base, top, module):
        if not isinstance(base, str) or not top:
            return False
        first = base.split(".")[0]
        if self.is_local(first):
            return False
        if top == first:
            return True
        return False

    ## Converge argument candidates under the receiver-preserving ufunc rule.
    #  @param candidate_groups One candidate-owner list per ufunc argument.
    #  @return Result owner or "unknown".
    def _receiver_preserving_ufunc_owner(self, candidate_groups):
        owners = []
        for candidates in candidate_groups:
            unique = self._dedupe_list(
                candidate for candidate in candidates
                if candidate not in (None, ""))
            if len(unique) != 1:
                return "unknown"
            owners.append(unique[0])
        if any(owner in ("local", "unknown") for owner in owners):
            return "unknown"
        external = self._dedupe_list(
            owner for owner in owners if owner != "python")
        if not external:
            return "numpy"
        if len(external) == 1 and external[0] in ("numpy", "pandas"):
            return external[0]
        return "unknown"

    ## Converge exact operands for a local arithmetic return expression.
    #  @param candidate_groups One candidate-owner list per expression operand.
    #  @return Result owner or "unknown".
    def _bounded_expression_owner(self, candidate_groups):
        owners = []
        for candidates in candidate_groups:
            unique = self._dedupe_list(
                candidate for candidate in candidates
                if candidate not in (None, ""))
            if len(unique) != 1 or unique[0] == "unknown":
                return "unknown"
            owners.append(unique[0])
        external = self._dedupe_list(
            owner for owner in owners
            if owner not in ("local", "python"))
        if len(external) == 1 and all(
                owner in (external[0], "python") for owner in owners):
            return external[0]
        if owners and all(owner == "python" for owner in owners):
            return "python"
        if owners and all(owner == "local" for owner in owners):
            return "local"
        return "unknown"

    ## Collect all origin candidates from a source value.
    #  @param module Current module name.
    #  @param source Source value to expand.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @param include_local Whether to include "local" in results.
    #  @return List of candidate top strings.
    def _origin_candidates(self, module, source, tracers, include_local=True,
                           _seen=None):
        if _seen is None:
            _seen = set()
        source = normalize_source(source)
        key = (module, type(source).__name__, source_display(source))
        if key in _seen:
            return ["unknown"]
        seen = set(_seen)
        seen.add(key)

        if isinstance(source, SourceSet):
            out = []
            for item in source.sources:
                out.extend(self._origin_candidates(
                    module, item, tracers, include_local,
                    _seen=set(seen)))
            return self._dedupe_list(out)
        if isinstance(source, PythonShape):
            return ["python"]
        if isinstance(source, DerivedResult):
            if source.kind == "iterator":
                return ["unknown"]
            if source.kind == "receiver_preserving_ufunc":
                candidate_groups = [
                    self._argument_owner_candidates(module, item, tracers)
                    for item in source.sources
                ]
                return [self._receiver_preserving_ufunc_owner(
                    candidate_groups)]
            if source.kind == "method_result":
                if len(source.sources) != 1:
                    return ["unknown"]
                method_source = normalize_source(source.sources[0])
                if not isinstance(method_source, InstanceMethod):
                    return ["unknown"]
                resolved = self._resolve_structured_source(
                    module, method_source, tracers, _seen=set(seen))
                if resolved is None:
                    return ["unknown"]
                _, receiver_module, receiver_symbol = resolved
                receiver_top = self._top_source(
                    receiver_module, receiver_symbol, tracers,
                    _seen=set(seen))
                if receiver_top == "python":
                    return ["python"]
                result_owner = _match_result_owner(
                    receiver_top, method_source.method)
                return [result_owner or "unknown"]
            if source.kind == "expression":
                # An expression receiver is owner evidence only when every
                # operand resolves to the same concrete owner.  A parameter
                # expression with unresolved or mixed operands must remain
                # unknown rather than falling through to local.  Explicit
                # local evidence is retained when no external owner is
                # present, which preserves local protocol receivers such as
                # ``(self.value * mask).sum()``.
                operand_candidates = []
                numeric_scalars = True
                saw_local = False
                saw_unresolved = False
                for item in source.sources:
                    shape = self._returned_python_shape(module, item, tracers)
                    candidates = (["python"] if shape is not None else
                                  self._argument_owner_candidates(
                                      module, item, tracers))
                    saw_local = saw_local or "local" in candidates
                    saw_unresolved = (
                        saw_unresolved or "unknown" in candidates)
                    concrete = self._dedupe_list([
                        candidate for candidate in candidates
                        if candidate not in (None, "", "unknown", "local")
                    ])
                    if len(concrete) != 1:
                        if concrete:
                            return ["unknown"]
                        saw_unresolved = True
                        continue
                    if concrete[0] == "python":
                        numeric_scalars = numeric_scalars and (
                            shape is not None and shape.kind in (
                                "bool", "int", "float", "complex"))
                    operand_candidates.append(concrete[0])
                external = set(operand_candidates) - {"python"}
                if (len(external) == 1 and "python" in operand_candidates
                        and numeric_scalars and not saw_unresolved
                        and not saw_local and source.attribute in (
                            "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod",
                            "Pow", "Compare")):
                    return [next(iter(external))]
                if (operand_candidates and not saw_unresolved
                        and not saw_local and all(
                        owner == operand_candidates[0]
                        for owner in operand_candidates)):
                    return [operand_candidates[0]]
                if not operand_candidates and saw_local:
                    return ["local"]
                return ["unknown"]
            out = []
            for item in source.sources:
                out.extend(self._origin_candidates(
                    module, item, tracers, include_local,
                    _seen=set(seen)))
            return self._dedupe_list(out) or ["unknown"]
        if isinstance(source, UnknownSource):
            return ["unknown"]
        if isinstance(source, ContainerItem):
            container = normalize_source(source.container)
            if (isinstance(container, CallResult)
                    and isinstance(source.index, int)):
                source_origin_module = container.source_module or module
                candidates = self._bounded_call_result_item_candidates(
                    source_origin_module, container, source.index, tracers,
                    _seen=seen)
                if candidates is not None:
                    return candidates
                # No project-local tuple contract exists. Preserve the
                # established aggregate call-result fallback for external
                # calls instead of treating the positional marker as a
                # project container lookup.
                return self._origin_candidates(
                    source_origin_module, container, tracers, include_local,
                    _seen=set(seen))
        if isinstance(source, ContainerIter):
            resolved = self._resolve_container_iter(
                module, source.container, tracers)
            if resolved is None:
                return ["unknown"]
            _, candidates = resolved
            return self._dedupe_list(candidates) or ["unknown"]
        if (isinstance(source, InstanceMethod)
                and isinstance(
                    normalize_source(source.receiver), CallResult)):
            return self._origin_candidates(
                module, normalize_source(source.receiver), tracers,
                include_local, _seen=set(seen))
        if isinstance(source, CallResult):
            source_origin_module = source.source_module or module
            if source.result_source is not None:
                if (isinstance(source.result_source, str)
                        and source.result_source not in (
                            "", "local", "python", "unknown")):
                    return [source.result_source]
                return self._origin_candidates(
                    source_origin_module, source.result_source, tracers,
                    include_local,
                    _seen=set(seen))
            bounded = self._bounded_call_result_candidates(
                module, source, tracers, _seen=set(seen))
            if bounded is not None:
                return bounded
            if self._local_class_from_source(module, source) is not None:
                return ["local"]
            callee = source.callee
            tracer = tracers.get(module)
            rs = tracer.return_sources.get(callee) if tracer else None
            if rs is not None:
                candidates = self._origin_candidates(
                    module, rs, tracers, include_local,
                    _seen=set(seen))
                clean = [c for c in candidates
                         if c not in ("", None, "unknown")]
                if clean:
                    return clean
                cr_lineno = getattr(source, 'call_lineno', 0) or 0
                cr_col = getattr(source, 'call_col_offset', 0) or 0
                if cr_lineno:
                    rs_norm = normalize_source(rs)
                    if isinstance(rs_norm, SourceSet):
                        for s in rs_norm.sources:
                            if isinstance(s, str):
                                arg = self._resolve_param_to_arg(
                                    module, callee, s, tracers,
                                    call_lineno=cr_lineno, call_col_offset=cr_col)
                                if arg is not None:
                                    more = self._origin_candidates(
                                        module, arg, tracers, include_local,
                                        _seen=set(seen))
                                    for m in more:
                                        if m not in candidates:
                                            candidates.append(m)
                    else:
                        arg = self._resolve_param_to_arg(
                            module, callee, rs, tracers,
                            call_lineno=cr_lineno, call_col_offset=cr_col)
                        if arg is not None:
                            more = self._origin_candidates(
                                module, arg, tracers, include_local,
                                _seen=set(seen))
                            for m in more:
                                if m not in candidates:
                                    candidates.append(m)
                return candidates
            top = self._top_source(
                source_origin_module, callee, tracers, _seen=set(seen))
            return [top] if top else []
        if is_structured_source(source):
            resolved = self._resolve_structured_source(
                module, source, tracers, _seen=set(seen))
            if resolved is not None:
                _, src_module, src_symbol = resolved
                return self._origin_candidates(
                    src_module, src_symbol, tracers, include_local,
                    _seen=set(seen))
            return ["unknown"]
        if isinstance(source, str):
            top = self._top_source(
                module, source, tracers, _seen=set(seen))
            return [top] if top else []
        return ["unknown"]

    ## Deduplicate a list preserving order.
    #  @param items List of strings.
    #  @return Deduplicated list.
    @staticmethod
    def _dedupe_list(items):
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    ## Determine the classification reason for a resolved API call.
    #  @param base The call's base symbol or source.
    #  @param top The resolved top-level library.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @return Reason constant string.
    # ── classification helpers ───────────────────────────────────────────

    ## Determine confidence for a classification result.
    #
    #  Delegates to the standalone classify_confidence() in
    #  classification.py so the confidence rules live in one place.
    #  @param reason Classification reason.
    #  @param alternatives List of alternative top libraries.
    #  @return Confidence score (0.0-1.0).
    def _classify_confidence(self, reason, alternatives=None):
        return classify_confidence(reason, alternatives)

    ## Unified classification entry point for a resolved top library.
    #
    #  Delegates to ClassificationPipeline.classify() (Phase 8B).
    #  Kept as a thin wrapper so callers in get_calls() and
    #  _build_symbol_provenance() do not need to change.
    #  @param base The call's base symbol or source.
    #  @param top The resolved top-level library.
    #  @param module Current module name.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param module_tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return ClassificationResult with library/reason/confidence/alternatives.
    def classify_source(self, base, top, module, tracer, module_tracers,
                        expand_origins=True, symbol=None, kind=""):
        if kind == "import" and self._is_direct_external_import(base, top, module):
            # Override: direct external imports always use DIRECT_IMPORT reason.
            result = self._pipeline.classify(
                base, top, module, tracer, module_tracers,
                expand_origins=expand_origins)
            return ClassificationResult(
                library=result.library,
                reason=REASON_DIRECT_IMPORT,
                confidence=classify_confidence(REASON_DIRECT_IMPORT),
                alternatives=result.alternatives,
                is_usage_library=result.is_usage_library)
        return self._pipeline.classify(
            base, top, module, tracer, module_tracers,
            expand_origins=expand_origins)

    ## Resolve the first segment of func_name to its fully qualified path.
    #  @param call_dict Dict with 'func_name' and other call data.
    #  @param module The module where the call occurs.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param _visited Set of already-visited first names (cycle detection).
    #  @return Resolved function path string.
    def _resolve_func_name(self, call_dict, module, tracer, _visited=None):
        func_name = call_dict.get('func_name', '')
        if not func_name:
            return func_name
        base = normalize_source(call_dict.get('base'))
        if isinstance(base, SuperMethod):
            base_path = call_dict.get('super_base_path')
            decorator_module = call_dict.get('super_decorator_module')
            decorator_parts = decorator_module.split('.') if decorator_module else []
            local_decorator = any(
                self.is_local('.'.join(decorator_parts[:length]))
                for length in range(1, len(decorator_parts) + 1)
            )
            if (isinstance(base_path, str) and base_path
                    and not self.is_local(base_path.split('.')[0])
                    and not local_decorator):
                # Public import provenance is a display hint, not the
                # method's runtime defining class or an importability claim.
                return base_path + '.' + base.method
            return func_name
        parts = func_name.split('.')
        first = parts[0]

        if _visited is None:
            _visited = set()
        if first in _visited:
            return func_name
        _visited.add(first)

        replacement = None
        if "call_import_source" in call_dict:
            ifs = call_dict["call_import_source"]
        else:
            ifs = tracer.import_from_symbols.get(first)
        if ifs:
            ifs_top = ifs.split('.')[0]
            if not self.is_local(ifs_top):
                replacement = ifs
        else:
            # 1.0.5 P0: prefer call-site snapshot (key present)
            # over final tracer map so RHS sub-calls see
            # pre-assignment state.  A snapshot of None means
            # "not in call_assign_funcs at call time", which
            # must not be replaced by a later assignment.
            if 'call_assign_func' in call_dict:
                caf = call_dict['call_assign_func']
            else:
                caf = tracer.call_assign_funcs.get(first)
            # Bare-name factories are represented directly by the call-result
            # source rather than call_assign_funcs (which records dotted
            # callees).  Use that exact source only when it identifies one
            # callee; SourceSet and other structured callees are intentionally
            # left unresolved instead of selecting one candidate.
            if not caf:
                call_base = normalize_source(call_dict.get('base'))
                if isinstance(call_base, CallResult):
                    factory_callee = normalize_source(call_base.callee)
                    if isinstance(factory_callee, str):
                        if (call_base.display_name
                                and '.' in call_base.display_name):
                            caf = call_base.display_name
                        else:
                            caf = factory_callee
            if caf and not caf.startswith(first + '.'):
                resolved_callee = self._resolve_func_name({'func_name': caf}, module, tracer, _visited)
                if resolved_callee and not resolved_callee.startswith('self.'):
                    replacement = resolved_callee

            if replacement is None:
                sd = tracer.symbols.direct.get(first)
                if isinstance(sd, str):
                    if sd == 'local' or sd == 'self' or sd.startswith('self.'):
                        return func_name
                    # If sd is a simple local name, try to resolve it further
                    if '.' not in sd:
                        gs_sd = self.global_symbols.get(module, {}).get(sd)
                        if gs_sd and gs_sd != 'local' and gs_sd != 'python':
                            replacement = gs_sd
                        else:
                            replacement = sd
                    else:
                        replacement = sd

            if replacement is None:
                gs = self.global_symbols.get(module, {}).get(first)
                if isinstance(gs, str):
                    if gs == 'local' or gs == 'python':
                        return func_name
                    replacement = gs
                elif gs is not None:
                    return func_name

        if replacement is None:
            return func_name

        # If the replacement's root is a local symbol, don't use it
        rep_first = replacement.split('.')[0]
        if rep_first == 'self' or (rep_first and rep_first != first):
            rep_gs = self.global_symbols.get(module, {}).get(rep_first)
            if rep_gs == 'local':
                return func_name

        if len(parts) == 1:
            return replacement

        # 1.0.5 P1: if the call's base has a class.method form (e.g.
        # InstanceMethod("Session","get")), resolve the class through
        # import_from_symbols for a more precise resolved_func
        # (e.g. requests.Session.get instead of requests.get).
        base_raw = call_dict.get('base', '')
        base_norm = normalize_source(base_raw)
        if isinstance(base_norm, InstanceMethod):
            receiver = base_norm.receiver
            if isinstance(receiver, str) and receiver in tracer.import_from_symbols:
                ifs = tracer.import_from_symbols[receiver]
                if not self.is_local(ifs.split('.')[0]):
                    return ifs + '.' + base_norm.method

        return replacement + '.' + '.'.join(parts[1:])

    ## Resolve cross-file symbol references across all modules.
    #
    #  For each symbol in each module, trace its source through imports
    #  and assignments to find the final origin.
    #  @param module_tracers Dict of module_name -> SingleFileAnalyzer.
    def resolve_cross_file_symbols(self, module_tracers):
        self._call_searched_global = set()
        for module, tracer in module_tracers.items():
            self.global_symbols[module] = {}
            self.symbol_chains[module] = {}
            for symbol, direct_source in tracer.symbols.direct.items():
                chain = self.trace_symbol(module, symbol, module_tracers, set())
                if chain:
                    chain = _dedup_consecutive(chain)
                    self.global_symbols[module][symbol] = self.extract_final_source(chain)
                    self.symbol_chains[module][symbol] = chain
        self._call_searched_global = None

    ## Normalize a container index to its positive equivalent.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param container_name Name of the container variable.
    #  @param key_idx Raw index (may be negative).
    #  @return Adjusted index.
    def _container_index(self, tracer, container_name, key_idx):
        if not isinstance(key_idx, int):
            return key_idx
        if key_idx >= 0:
            return key_idx
        n = tracer.container_lengths.get(container_name)
        if n is not None:
            return key_idx + n
        return key_idx

    ## Resolve a container item access to its source symbol.
    #
    #  Looks up the item in the current module's container_items, and falls
    #  back to cross-file import if not found locally.
    #  @param module The module where the access occurs.
    #  @param container_name Name of the container variable.
    #  @param key_idx The index/key being accessed.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return (src_module, src_symbol) tuple, or None.
    def _resolve_container_item(self, module, container_name, key_idx, tracers):
        tracer = tracers.get(module)
        if not tracer:
            return None
        container_idx = self._container_index(tracer, container_name, key_idx)
        item_key = (container_name, container_idx)
        if item_key in tracer.container_items:
            return (module, tracer.container_items[item_key])
        container_direct = tracer.symbols.direct.get(container_name)
        if self.is_local(container_direct):
            src_module = container_direct
            src_tracer = tracers.get(src_module)
            if not src_tracer:
                return None
            container_idx_src = self._container_index(src_tracer, container_name, key_idx)
            src_key = (container_name, container_idx_src)
            if src_key in src_tracer.container_items:
                return (src_module, src_tracer.container_items[src_key])
        return None

    ## Add a candidate to the list if not already visited.
    #  @param module The current module.
    #  @param src The source symbol.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @param candidates List to append to.
    #  @param visited Set of already-visited origins.
    def _container_candidate(self, module, src, tracers, candidates, visited):
        if not src:
            return
        top_src = self._top_source(module, src, tracers)
        if top_src and top_src not in visited:
            visited.add(top_src)
            candidates.append(top_src)

    ## Collect all candidates for a container's iteration source.
    #  @param module The current module.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param container_name Name of the container variable.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return List of candidate source symbols.
    def _collect_container_candidates(self, module, tracer, container_name, tracers):
        candidates = []
        visited = set()
        for (cont_name, idx), src in tracer.container_items.items():
            if cont_name == container_name:
                self._container_candidate(module, src, tracers, candidates, visited)
        for src in sorted(
                tracer.container_set_sources.get(container_name, set()),
                key=source_display):
            self._container_candidate(module, src, tracers, candidates, visited)
        return candidates

    ## Resolve Python element shapes carried by an iterable argument.
    #
    #  This uses only a concrete PythonShape already recorded at a call site.
    #  A dictionary's value shape is not reused for iteration because Python
    #  iteration yields keys, not values. Unknown or mixed shapes remain
    #  unknown.
    #  @param source Argument source or PythonShape.
    #  @return List of element sources, or None when no Python shape was
    #  proven for this argument.
    def _python_iterable_element_sources(self, source):
        source = normalize_source(source)
        if isinstance(source, PythonShape):
            if source.kind == "str":
                return [source]
            if source.kind in ("list", "tuple", "set") and source.item_kind:
                return [PythonShape(source.item_kind)]
            return None
        if isinstance(source, SourceSet):
            candidates = []
            for item in source.sources:
                item_sources = self._python_iterable_element_sources(item)
                if item_sources is None:
                    return None
                candidates.extend(item_sources)
            return self._dedupe_list(candidates) or ["unknown"]
        return None

    ## Resolve a concrete Python protocol through exact local return contexts.
    #  @param module Source module.
    #  @param source Value source, distinct from its primary owner.
    #  @param tracers Per-module analyzers.
    #  @param context Enclosing local call context, when available.
    #  @param seen Recursion guard.
    #  @return Uniform PythonShape, or None for mixed or unsupported values.
    def _returned_python_shape(self, module, source, tracers,
                               context=None, seen=None):
        key = (module, repr(source),
               (context.caller_module, context.edge.call_lineno,
                context.edge.call_col_offset, context.target)
               if context is not None else None)
        if key in self._python_shape_in_progress:
            return None
        self._python_shape_in_progress.add(key)
        try:
            return self._infer_returned_python_shape(
                module, source, tracers, context, seen)
        finally:
            self._python_shape_in_progress.remove(key)

    ## Infer a Python shape while the cross-resolver recursion guard is held.
    #  @param module Source module.
    #  @param source Value source.
    #  @param tracers Per-module analyzers.
    #  @param context Exact local call context.
    #  @param seen Path-local recursion guard.
    #  @return Proven PythonShape or None.
    def _infer_returned_python_shape(self, module, source, tracers,
                                     context=None, seen=None):
        source = normalize_source(source)
        if isinstance(source, PythonShape):
            return source
        key = ("python-return-shape", module, repr(source),
               (context.caller_module, context.edge.call_lineno,
                context.edge.call_col_offset, context.target)
               if context is not None else None)
        seen = set(seen or set())
        if key in seen:
            return None
        seen.add(key)

        def uniform(values):
            result = None
            for value in values:
                if value is None or (result is not None and value != result):
                    return None
                result = value
            return result

        if isinstance(source, SourceSet):
            return uniform(
                self._returned_python_shape(module, item, tracers, context, seen)
                for item in source.sources)
        if (isinstance(source, DerivedResult) and source.kind == 'method_result'
                and len(source.sources) == 1):
            return self._returned_python_shape(
                module, source.sources[0], tracers, context, seen)
        if isinstance(source, ParameterSource):
            if source.attributes or (source.derived
                                    and source.derived_operation != "slice"):
                return None
            current = context
            while current is not None:
                if (module == current.target.module
                        and source.scope == current.target.qualname):
                    summary = self.project_cg.modules[module].functions[
                        source.scope]
                    if source.name not in summary.params:
                        return None
                    arguments = self._edge_parameter_sources(
                        current.edge, summary, source.name,
                        summary.params.index(source.name),
                        prefer_protocol_shape=True)
                    shape = uniform(
                        self._returned_python_shape(
                            current.caller_module, argument, tracers,
                            current.parent, seen)
                        for argument in arguments or [])
                    break
                current = current.parent
            else:
                tracer = tracers.get(module)
                params = (tracer.function_params.get(source.scope, [])
                          if tracer is not None else [])
                if source.name not in params:
                    return None
                arguments = self._parameter_call_arguments(
                    module, source.scope, source.name, params.index(source.name),
                    tracer, tracers, prefer_protocol_shape=True)
                shape = uniform(
                    self._returned_python_shape(origin, argument, tracers,
                                                None, seen)
                    for origin, argument in arguments)
            if source.derived and (shape is None or shape.kind not in (
                    "str", "bytes", "list", "tuple")):
                return None
            return shape
        if isinstance(source, CallResult):
            origin = source.source_module or module
            contexts = self._bounded_call_contexts(
                origin, source.call_lineno, source.call_col_offset, tracers,
                parent=context, callee_name=source.display_name)
            if contexts:
                shapes = []
                for called in contexts:
                    if (called.target.qualname.endswith(".__init__")
                            and not called.edge.callee_name.endswith(".__init__")):
                        return None
                    summary = self.project_cg.modules[
                        called.target.module].functions[called.target.qualname]
                    shapes.append(self._returned_python_shape(
                        called.target.module, summary.return_values,
                        tracers, called, seen))
                    if shapes[-1] is None or shapes[-1] != shapes[0]:
                        return None
                return uniform(shapes)
            if source.result_source is not None:
                return self._returned_python_shape(
                    origin, source.result_source, tracers, context, seen)
            return None
        if isinstance(source, InstanceMethod):
            receiver = source.receiver
            if (source.parameter_scope and source.parameter_name
                    and receiver == source.parameter_name):
                receiver = ParameterSource(source.parameter_scope,
                                           source.parameter_name)
            shape = self._returned_python_shape(
                module, receiver, tracers, context, seen)
            result = _builtin_method_return_shape(shape, source.method)
            if result is not None:
                return result
            # Reuse an already verified external result contract only.
            if not _has_result_owner_contract(source.method):
                return None
            owners = self._origin_candidates(module, receiver, tracers)
            if len(owners) == 1:
                return _match_result_python_shape(owners[0], source.method)
        if isinstance(source, ContainerItem):
            shape = self._returned_python_shape(
                module, source.container, tracers, context, seen)
            if shape is not None:
                if shape.kind == "str":
                    return PythonShape("str")
                if shape.kind in ("list", "tuple") and shape.item_kind:
                    return PythonShape(shape.item_kind)
        return None

    ## Resolve project-returned elements without erasing their value sources.
    #  @param module Module containing the source expression.
    #  @param source Element source, or iterable when iterable is True.
    #  @param tracers Per-module analyzers.
    #  @param context Exact return-call context, if available.
    #  @param iterable Whether to select elements from this source.
    #  @param _seen Recursion guard.
    #  @return (module, source) pairs, or None without an iterable contract.
    def _returned_element_sources(self, module, source, tracers,
                                  context=None, iterable=False, _seen=None):
        source = normalize_source(source)
        context_key = ((context.caller_module, context.edge.call_lineno,
                        context.edge.call_col_offset, context.target)
                       if context is not None else None)
        key = ("returned-element", module, source_display(source),
               context_key, iterable)
        seen = set(_seen or set())
        if key in seen:
            return [(module, UnknownSource("recursive returned iterable"))]
        seen.add(key)

        if isinstance(source, SourceSet):
            elements = []
            for branch in source.sources:
                elements.extend(self._returned_element_sources(
                    module, branch, tracers, context, iterable, seen)
                    or [(module, UnknownSource("unresolved return branch"))])
            return elements
        if isinstance(source, ContainerIter):
            return self._returned_element_sources(
                module, source.container, tracers, context, True, seen)
        if isinstance(source, ParameterSource):
            if source.derived or source.attributes:
                return None
            current = context
            while current is not None:
                if (module == current.target.module
                        and source.scope == current.target.qualname):
                    summary = self.project_cg.modules[
                        module].functions[current.target.qualname]
                    if source.name not in summary.params:
                        return None
                    arguments = self._edge_parameter_sources(
                        current.edge, summary, source.name,
                        summary.params.index(source.name),
                        prefer_protocol_shape=True)
                    elements = []
                    for argument in arguments or [UnknownSource()]:
                        elements.extend(self._returned_element_sources(
                            current.caller_module, argument, tracers,
                            current.parent, iterable, seen)
                            or [(current.caller_module, UnknownSource())])
                    return elements
                current = current.parent
            tracer = tracers.get(module)
            params = (tracer.function_params.get(source.scope, [])
                      if tracer is not None else [])
            if source.name not in params:
                return None
            arguments = self._parameter_call_arguments(
                module, source.scope, source.name, params.index(source.name),
                tracer, tracers, prefer_protocol_shape=True)
            elements = []
            for argument_module, argument in arguments:
                resolved = self._returned_element_sources(
                    argument_module, argument, tracers, None, iterable, seen)
                if resolved is None:
                    # The ordinary parameter-iteration path also retains
                    # literal element facts on call edges. Let it handle
                    # arguments not described by a return context here.
                    return None
                elements.extend(resolved)
            return elements or None
        if not iterable:
            return [(module, source)]
        shape = self._returned_python_shape(module, source, tracers, context)
        if shape is not None:
            shapes = self._python_iterable_element_sources(shape)
            if shapes is not None:
                return [(module, item) for item in shapes]
        shapes = self._python_iterable_element_sources(source)
        if shapes is not None:
            return [(module, shape) for shape in shapes]
        if isinstance(source, TupleSource):
            return [(module, item) for item in source.items]
        if isinstance(source, CallResult):
            origin = source.source_module or module
            contexts = self._bounded_call_contexts(
                origin, source.call_lineno, source.call_col_offset,
                tracers, parent=context, callee_name=source.display_name)
            if not contexts:
                shapes = self._python_iterable_element_sources(
                    source.result_source)
                return ([(origin, shape) for shape in shapes]
                        if shapes is not None else None)
            elements = []
            for called in contexts:
                tracer = tracers[called.target.module]
                summary = self.project_cg.modules[
                    called.target.module].functions[called.target.qualname]
                # A generator's return value terminates iteration; only its
                # yield summary describes values observed by a for-loop.
                returned = ([summary.yields] if summary.yields is not None
                            else tracer.return_element_sources.get(
                                called.target.qualname))
                for element in returned or [UnknownSource()]:
                    elements.extend(self._returned_element_sources(
                        called.target.module, element, tracers, called,
                        False, seen)
                        or [(called.target.module, UnknownSource())])
            return elements
        return None

    ## Resolve an iteration over a container to its source(s).
    #  @param module The current module.
    #  @param container_name Name of the container variable.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return (src_module, candidates_list) tuple, or None.
    def _resolve_container_iter(self, module, container_name, tracers):
        tracer = tracers.get(module)
        if not tracer:
            return None
        elements = self._returned_element_sources(
            module, container_name, tracers, iterable=True)
        if elements is not None:
            candidates = []
            for origin, element in elements:
                candidates.extend(self._origin_candidates(
                    origin, element, tracers, include_local=True))
            return (module, self._dedupe_list(candidates) or ["unknown"])
        if not isinstance(container_name, str):
            cn = normalize_source(container_name)
            if isinstance(cn, ParameterSource):
                if cn.derived or cn.attributes:
                    return (module, ["unknown"])
                params = (
                    tracer.function_params.get(cn.scope)
                    or tracer.function_params.get(
                        cn.scope.rsplit(".", 1)[-1], []))
                if cn.name not in params:
                    return (module, ["unknown"])
                param_index = params.index(cn.name)
                protocol_arguments = self._parameter_call_arguments(
                    module, cn.scope, cn.name, param_index,
                    tracer, tracers, prefer_protocol_shape=True)
                if protocol_arguments:
                    element_sources = []
                    protocol_proven = True
                    for _, argument in protocol_arguments:
                        argument_sources = (
                            self._python_iterable_element_sources(argument))
                        if argument_sources is None:
                            protocol_proven = False
                            break
                        element_sources.extend(argument_sources)
                    if protocol_proven and element_sources:
                        return (
                            module,
                            self._dedupe_list(element_sources) or ["unknown"],
                        )
                arguments = self._parameter_call_arguments(
                    module, cn.scope, cn.name, param_index,
                    tracer, tracers, prefer_iterable_elements=True)
                if not arguments:
                    return (module, ["unknown"])
                candidates = []
                for caller_module, source in arguments:
                    candidates.extend(self._origin_candidates(
                        caller_module, source, tracers,
                        include_local=True))
                return (
                    module,
                    self._dedupe_list(candidates) or ["unknown"],
                )
            if isinstance(cn, CallResult) and isinstance(cn.callee, str):
                elements = tracer.return_element_sources.get(cn.callee)
                if elements is not None:
                    candidates = []
                    for element in elements:
                        candidates.extend(self._origin_candidates(
                            module, element, tracers, include_local=True))
                    return (module, self._dedupe_list(candidates) or ["unknown"])
                top = self._top_source(module, cn.callee, tracers)
                if top and top not in ("local", "python", "unknown", ""):
                    ## 1.0.5 P2: only propagate callee top as element type
                    #  when the callee has explicit return-type evidence.
                    #  Without return_sources, an import-backed call result
                    #  has no yield contract — element type is unknowable.
                    if tracer.return_sources.get(cn.callee) is not None:
                        return (module, [top])
                ## No yield contract or builtin callee:
                #  element type cannot be determined statically.
                return (module, ["unknown"])
            return (module, ["unknown"])
        local_candidates = self._collect_container_candidates(module, tracer, container_name, tracers)
        if local_candidates:
            return (module, local_candidates)
        container_direct = tracer.symbols.direct.get(container_name)
        if isinstance(container_direct, str) and self.is_local(container_direct):
            src_module = container_direct
            src_tracer = tracers.get(src_module)
            if not src_tracer:
                return None
            src_candidates = self._collect_container_candidates(src_module, src_tracer, container_name, tracers)
            if src_candidates:
                return (src_module, src_candidates)
        return None

    ## Resolve a method call through class inheritance and cross-file imports.
    #
    #  Searches the class's method list, then recursively checks parent classes,
    #  following imports to other modules as needed.
    #  @param module The current module.
    #  @param class_symbol The class name.
    #  @param method_name The method being called.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @param visited Set of already-visited (module, class, method) keys.
    #  @return (src_module, src_symbol) tuple, or None.
    def _resolve_method_symbol(self, module, class_symbol, method_name, tracers, visited):
        tracer = tracers.get(module)
        if not tracer:
            return None
        key = (module, class_symbol, method_name)
        if key in visited:
            return None
        visited.add(key)
        methods = tracer.class_methods.get(class_symbol, [])
        if method_name in methods:
            return (module, method_name)
        for base_symbol in tracer.class_bases.get(class_symbol, []):
            if base_symbol in tracer.class_methods:
                resolved = self._resolve_method_symbol(module, base_symbol, method_name, tracers, visited)
                if resolved:
                    return resolved
            base_direct = normalize_source(tracer.symbols.direct.get(base_symbol))
            if isinstance(base_direct, CallResult):
                base_direct = base_direct.callee
                if base_direct == base_symbol:
                    base_direct = tracer.import_from_symbols.get(base_symbol, base_direct)
            if isinstance(base_direct, str):
                if self.is_local(base_direct):
                    src_module = base_direct
                    resolved = self._resolve_method_symbol(src_module, base_symbol, method_name, tracers, visited)
                    if resolved:
                        return resolved
                else:
                    return (module, base_symbol)
            # P0: handle external base classes not in symbols.direct.
            # When a local class inherits from tornado.tcpserver.TCPServer
            # and the base_symbol dotted name isn't stored as a symbol,
            # check whether the top-level prefix is an import.
            if (isinstance(base_symbol, str) and '.' in base_symbol
                    and not base_direct):
                prefix = base_symbol.split('.')[0]
                aliases = getattr(tracer, "import_aliases", set())
                alias_prefixes = {
                    a.split('.')[0] for a in aliases if isinstance(a, str)}
                alias_prefixes |= {
                    a.split('.')[0] for a in getattr(
                        tracer, "import_from_symbols", {}) if isinstance(a, str)}
                if prefix in alias_prefixes:
                    return (module, prefix)
        class_direct = normalize_source(tracer.symbols.direct.get(class_symbol))
        if isinstance(class_direct, CallResult):
            class_direct = class_direct.callee
            if class_direct == class_symbol:
                class_direct = tracer.import_from_symbols.get(class_symbol, class_direct)
        if isinstance(class_direct, str):
            if self.is_local(class_direct):
                src_module = class_direct
                resolved = self._resolve_method_symbol(src_module, class_symbol, method_name, tracers, visited)
                if resolved:
                    return resolved
            else:
                if class_direct == "local":
                    rs = tracer.return_sources.get(class_symbol)
                    if rs is not None and isinstance(rs, tuple) and len(rs) == 3 and rs[0] == "call_result":
                        return (module, rs[1])
                return (module, class_symbol)
        return None

    ## Trace a function or constructor parameter through collected call sites.
    #  @param module Module containing the parameter.
    #  @param param_name Parameter name to resolve.
    #  @param display_symbol Symbol to use at the start of the returned chain.
    #  @param tracer Single-file analyzer for module.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @param visited Set of visited trace keys.
    #  @return Chain from display_symbol to argument origin, or None.
    def _trace_parameter_source(self, module, param_name, display_symbol, tracer, tracers, visited):
        for func_name, params in tracer.function_params.items():
            try:
                param_idx = params.index(param_name)
            except ValueError:
                continue
            for call_site in tracer.call_sites.get(func_name, []):
                if param_idx >= len(call_site["args"]):
                    continue
                arg_src = call_site["args"][param_idx]
                if isinstance(arg_src, str):
                    sub_chain = self.trace_symbol(
                        call_site["module"], arg_src, tracers, visited
                    )
                elif arg_src is not None:
                    sub_chain = self.trace_symbol(
                        call_site["module"], param_name, tracers, set(),
                        _direct_source=arg_src,
                    )
                else:
                    sub_chain = None
                if sub_chain:
                    if sub_chain[0] == display_symbol:
                        return sub_chain
                    return [display_symbol] + sub_chain
        return None

    ## Resolve a structured (tuple) source to its concrete origin.
    #
    #  Handles the four structured tuple kinds:
    #  - "container_item" for subscript access
    #  - "instance_method" for method calls
    #  - "container_iter" for iteration over containers
    #  - "call_result" for function call return values
    #  @param module The current module.
    #  @param direct_source The structured tuple (kind, arg1, arg2).
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return (display_name, src_module, src_symbol) tuple, or None.

    ## Resolve a SourceSet to a primary top library using origin-aware rules.
    #
    #  Delegates to SourceSetResolver.  See source_resolution.py for the
    #  origin-aware convergence rules (dict_lookup, return, default).
    def _resolve_sourceset_primary(self, module, sourceset, tracers, _seen=None):
        return self._source_resolver.resolve_primary(
            module, sourceset, tracers, _seen=_seen)

    # ── structured source resolution ─────────────────────────────────────

    def _resolve_structured_source(self, module, direct_source, tracers,
                                   _seen=None):
        if _seen is None:
            _seen = set()
        direct_source = normalize_source(direct_source)
        structured_key = (
            "structured", module, type(direct_source).__name__,
            source_display(direct_source),
        )
        if structured_key in _seen:
            return (source_display(direct_source), module, "unknown")
        _seen = set(_seen)
        _seen.add(structured_key)
        if isinstance(direct_source, PythonShape):
            return (source_display(direct_source), module, "python")
        if isinstance(direct_source, DerivedResult):
            candidates = self._origin_candidates(
                module, direct_source, tracers, _seen=set(_seen))
            unique = self._dedupe_list([
                candidate for candidate in candidates
                if candidate not in (None, "")
            ])
            owner = unique[0] if len(unique) == 1 else "unknown"
            return (source_display(direct_source), module, owner)
        if isinstance(direct_source, SourceSet):
            ## 7B-full PR7-final: converge-check all candidates.
            primary = self._resolve_sourceset_primary(
                module, direct_source, tracers, _seen=set(_seen))
            if primary:
                return (source_display(direct_source), module, primary)
            ## Function branch values and element-derived builtins identify
            # multiple runtime possibilities. Preserve every origin as
            # alternatives without selecting one branch as primary.
            if direct_source.origin in (
                    "builtin_element", "function_branch"):
                return (source_display(direct_source), module, "unknown")
            ## Legacy module-level branch imports retain their compatibility
            # primary until a full module CFG replaces this fallback.
            for src in direct_source.sources:
                if isinstance(src, str):
                    top = self._top_source(
                        module, src, tracers, _seen=set(_seen))
                    if top and top not in (
                            "local", "python", "unknown", ""):
                        return (source_display(direct_source), module, src)
            for src in direct_source.sources:
                if isinstance(src, str):
                    return (source_display(direct_source), module, src)
            return None
        if isinstance(direct_source, ParameterSource):
            owners = self._dedupe_list([
                owner for owner in self._argument_owner_candidates(
                    module, direct_source, tracers)
                if owner not in (None, "")
            ])
            owner = owners[0] if len(owners) == 1 else "unknown"
            return (source_display(direct_source), module, owner)
        callee_display = None
        if isinstance(direct_source, ContainerItem):
            kind, a, b = "container_item", direct_source.container, direct_source.index
        elif isinstance(direct_source, ContainerIter):
            kind, a, b = "container_iter", direct_source.container, "*"
        elif isinstance(direct_source, InstanceMethod):
            kind, a, b = "instance_method", direct_source.receiver, direct_source.method
        elif isinstance(direct_source, SuperMethod):
            kind, a, b = "super_method", direct_source.class_key, direct_source.method
        elif isinstance(direct_source, CallResult):
            kind, a, b = "call_result", direct_source.callee, None
            callee_display = direct_source.display_name or direct_source.callee
        elif isinstance(direct_source, tuple) and len(direct_source) == 3:
            kind, a, b = direct_source
        else:
            return None

        if kind == "container_item":
            if isinstance(a, TupleSource):
                selected = _tuple_source_item(a, b)
                if selected is not None:
                    structured = self._resolve_structured_source(
                        module, selected, tracers, _seen=set(_seen))
                    if structured is not None:
                        _, src_module, src_symbol = structured
                        return (f"{source_display(a)}[{b}]",
                                src_module, src_symbol)
            if isinstance(a, ParameterSource):
                tracer = tracers.get(module)
                params = (tracer.function_params.get(a.scope)
                          if tracer is not None else None)
                if params is None and tracer is not None:
                    params = tracer.function_params.get(
                        a.scope.rsplit(".", 1)[-1], [])
                if tracer is not None and a.name in (params or []):
                    arguments = self._parameter_call_arguments(
                        module, a.scope, a.name, params.index(a.name),
                        tracer, tracers, prefer_iterable_elements=True)
                    selected_sources = []
                    for _, argument in arguments:
                        selected = _tuple_source_item(argument, b)
                        if selected is not None:
                            selected_sources.append(selected)
                    selected_sources = self._dedupe_list(
                        selected_sources)
                    if len(selected_sources) == 1:
                        structured = self._resolve_structured_source(
                            module, selected_sources[0], tracers,
                            _seen=set(_seen))
                        if structured is not None:
                            _, src_module, src_symbol = structured
                            return (f"{source_display(a)}[{b}]",
                                    src_module, src_symbol)
            resolved = self._resolve_container_item(module, a, b, tracers)
            if resolved:
                src_module, src_symbol = resolved
                return (f"{a}[{b}]", src_module, src_symbol)
            if is_structured_source(a):
                structured = self._resolve_structured_source(
                    module, a, tracers, _seen=set(_seen))
                if structured is not None:
                    _, src_module, src_symbol = structured
                    top = self._top_source(
                        src_module, src_symbol, tracers, _seen=set(_seen))
                    if top and top not in ("local", "unknown", ""):
                        return (f"{a}[{b}]", src_module, top)
            return (f"{a}[{b}]", module, a)

        if kind == "instance_method":
            tracer = tracers.get(module)
            if not tracer:
                return None
            if (isinstance(direct_source, InstanceMethod)
                    and isinstance(
                        direct_source.receiver, InstanceAttribute)):
                attribute_top = self._resolve_instance_attribute_method_top(
                    module, direct_source.receiver, direct_source.method,
                    tracers, _seen=set(_seen))
                return (
                    "%s.%s" % (
                        source_display(direct_source.receiver),
                        direct_source.method),
                    module,
                    attribute_top,
                )
            if (isinstance(direct_source, InstanceMethod)
                    and direct_source.parameter_scope):
                parameter_top = self._resolve_parameter_method_top(
                    module, direct_source, tracer, tracers,
                    _seen=set(_seen))
                return (
                    f"{source_display(a)}.{b}",
                    module,
                    parameter_top,
                )
            if isinstance(a, ContainerIter):
                candidates = self._argument_method_owner_candidates(
                    module, a, b, tracers, _seen=set(_seen))
                return (f"{source_display(a)}.{b}", module,
                        self._bounded_candidates_top(candidates))
            if isinstance(a, ParameterSource) and a.derived:
                candidates = self._argument_method_owner_candidates(
                    module, a, b, tracers, _seen=set(_seen))
                return (f"{source_display(a)}.{b}", module,
                        self._bounded_candidates_top(candidates))
            if (isinstance(direct_source, InstanceMethod)
                    and isinstance(direct_source.receiver, DerivedResult)
                    and self._is_bounded_expression_source(
                        direct_source.receiver)):
                parameter_top = self._resolve_derived_expression_method_top(
                    module, direct_source.receiver, direct_source.method,
                    tracer, tracers)
                return (
                    f"{source_display(a)}.{b}",
                    module,
                    parameter_top,
                )
            if (isinstance(direct_source, InstanceMethod)
                    and isinstance(direct_source.receiver, DerivedResult)):
                expression_top = self._resolve_expression_external_top(
                    module, direct_source.receiver, tracers)
                if (expression_top != "unknown"
                        or self._has_expression_owner_evidence(
                            direct_source.receiver)):
                    return (
                        f"{source_display(a)}.{b}",
                        module,
                        expression_top,
                    )
            if (isinstance(a, InstanceMethod)
                    and isinstance(a.receiver, str)
                    and _is_verified_result_owner(a.receiver)):
                return (
                    f"{source_display(a)}.{b}",
                    module,
                    a.receiver,
                )
            if isinstance(a, (CallResult, SourceSet)):
                local_classes = self._local_class_candidates(
                    module, a, tracers, visited=set(_seen))
                branch_evidence = isinstance(a, CallResult)
                if isinstance(a, SourceSet):
                    branch_evidence = bool(a.sources) and all(
                        self._local_class_candidates(
                            module, item, tracers, visited=set(_seen))
                        for item in a.sources
                    )
                if branch_evidence:
                    local_method_classes = [
                        identity for identity in local_classes
                        if self._local_class_defines_method(
                            identity[0], identity[1], b, tracers)
                    ]
                    if len(local_method_classes) == 1:
                        return (f"{source_display(a)}.{b}", module, "local")
                    if len(local_method_classes) > 1:
                        return (f"{source_display(a)}.{b}", module, "unknown")
                if isinstance(a, CallResult) and isinstance(a.callee, str):
                    for target_module, module_cg in self.project_cg.modules.items():
                        prefix = target_module + "."
                        if a.callee.startswith(prefix):
                            qualname = a.callee[len(prefix):]
                        elif (target_module == module
                              and a.callee in module_cg.functions):
                            qualname = a.callee
                        else:
                            continue
                        summary = module_cg.functions.get(qualname)
                        if (summary is not None
                                and self._has_unresolved_method_result(
                                    summary.returns)):
                            return (
                                f"{source_display(a)}.{b}",
                                module,
                                "unknown",
                            )
            if is_structured_source(a):
                receiver = self._resolve_structured_source(
                    module, a, tracers, _seen=set(_seen))
                if receiver is None:
                    return (f"{source_display(a)}.{b}", module, "unknown")
                _, receiver_module, receiver_symbol = receiver
                explicit_owner = (
                    a.result_source if isinstance(a, CallResult) else None)
                if (isinstance(explicit_owner, str)
                        and explicit_owner not in (
                            "", "local", "python", "unknown")):
                    receiver_top = explicit_owner
                else:
                    receiver_top = self._top_source(
                        receiver_module, receiver_symbol, tracers,
                        _seen=set(_seen))
                # A local callable identity does not prove the type of its
                # returned object.  Preserve ``local`` only when the call
                # result resolves to one project-local class that actually
                # defines this method; otherwise the receiver owner is
                # unresolved rather than project-local.
                if (receiver_top == "local"
                        and isinstance(a, CallResult)
                        and not local_classes):
                    receiver_top = "unknown"
                if (receiver_top == "local"
                        and isinstance(a, InstanceMethod)
                        and isinstance(a.receiver, InstanceMethod)
                        and a.receiver.parameter_scope):
                    receiver_top = "unknown"
                if not receiver_top:
                    receiver_top = "unknown"
                return (f"{source_display(a)}.{b}",
                        receiver_module, receiver_top)
            # A dynamically indexed homogeneous mapping can be resolved only
            # after the complete file has been visited.  This post-pass keeps
            # the evidence generic: every statically resolved value must
            # converge to one owner before it is used for the method call.
            if isinstance(a, str):
                homogeneous = getattr(
                    tracer, "homogeneous_container_value_sources", {}
                ).get(a)
                if homogeneous is not None:
                    resolved = self._resolve_structured_source(
                        module, homogeneous, tracers, _seen=set(_seen))
                    if resolved is not None:
                        _, receiver_module, receiver_symbol = resolved
                        receiver_top = self._top_source(
                            receiver_module, receiver_symbol, tracers,
                            _seen=set(_seen))
                        if receiver_top not in (None, "local", "python", "unknown", ""):
                            return (f"{a}.{b}", receiver_module, receiver_top)
            if a in tracer.import_from_symbols:
                class_symbol = a
            else:
                class_symbol = tracer.symbols.direct.get(a)
                if isinstance(class_symbol, tuple) and len(class_symbol) == 3 and class_symbol[0] == "call_result":
                    class_symbol = class_symbol[1]
                class_symbol = normalize_source(class_symbol)
                if isinstance(class_symbol, CallResult):
                    class_symbol = class_symbol.callee
                if a in tracer.class_methods and class_symbol == "local":
                    class_symbol = a
            # 1.0.5 P1: unified receiver object ownership.
            receiver_top = self._resolve_receiver_object_top(
                module, a, tracer, tracers)
            if receiver_top:
                return (f"{a}.{b}", module, receiver_top)
            if not class_symbol:
                if isinstance(a, str) and _is_builtin(a):
                    return (f"{a}.{b}", module, "python")
                if isinstance(a, str) and _is_import_origin(tracer, a):
                    return (f"{a}.{b}", module, a)
                # No factory top: trace through parameter sources
                # using a hardcoded "local" starting point.
                # Must NOT use call_assign_funcs — those carry
                # file-level final state that may include the
                # current call itself (e.g. s = s.strip()).
                receiver_chain = self.trace_symbol(
                    module, a, tracers, set(), _direct_source="local")
                receiver_top = self.extract_final_source(receiver_chain)
                if receiver_top:
                    return (f"{a}.{b}", module, receiver_top)
                return None
            # P0: if the method is explicitly defined in a local class,
            # preserve its primary identity as local.  The method body
            # may call library APIs internally, but the callable itself
            # is project-local.
            if class_symbol in tracer.class_methods:
                if b in tracer.class_methods[class_symbol]:
                    return (f"{a}.{b}", module, "local")
                ext = self._resolve_local_method_to_external(
                    module, class_symbol, b, a, tracer, tracers)
                if ext:
                    return (f"{a}.{b}", module, ext)
            resolved = self._resolve_method_symbol(module, class_symbol, b, tracers, set())
            if not resolved:
                if class_symbol != a and isinstance(a, str) and a in tracer.symbols.direct:
                    resolved = self._resolve_method_symbol(module, a, b, tracers, set())
                if not resolved:
                    class_direct = tracer.symbols.direct.get(class_symbol)
                    if isinstance(class_direct, str) and self.is_local(class_direct):
                        cg_attr = self._lookup_cg_class_attr_source(module, class_symbol, b)
                        if cg_attr is not None:
                            cg_mod, cg_top = cg_attr
                            return (f"{a}.{b}", cg_mod, cg_top)
                        return (f"{a}.{b}", module, "local")
                    if class_direct == "local":
                        ext = self._resolve_local_method_to_external(
                            module, a, b, a, tracer, tracers)
                        if ext:
                            return (f"{a}.{b}", module, ext)
                        cg_attr = self._lookup_cg_class_attr_source(module, class_symbol, b)
                        if cg_attr is not None:
                            cg_mod, cg_top = cg_attr
                            return (f"{a}.{b}", cg_mod, cg_top)
                        return (f"{a}.{b}", module, "local")
                    if isinstance(class_symbol, str) and '.' in class_symbol:
                        top = self._top_source(
                            module, class_symbol, tracers, _seen=set(_seen))
                        if top and top not in ("local", "python", "unknown", ""):
                            return (f"{a}.{b}", module, top)
                    # 1.0.5 P1: builtin container method on receiver
                    # whose container kind is known from the tracer.
                    # This fallback only sees module-level legacy maps.
                    # Function-local receiver kinds are carried by the
                    # call-site snapshot from single_file.
                    kind = _receiver_container_kind(tracer, a)
                    if (kind is not None
                            and b in _BUILTIN_CONTAINER_METHODS.get(kind, frozenset())
                            and a in tracer.symbols.direct):
                        return (f"{a}.{b}", module, "python")
                    return None
            src_module, src_symbol = resolved
            ## 7B-full PR3: if the resolved method is local, try constructor attrs.
            top = self._top_source(
                src_module, src_symbol, tracers, _seen=set(_seen))
            if top in ("local", "unknown", ""):
                cg_attr = self._lookup_cg_class_attr_source(
                    module, class_symbol, b)
                if cg_attr is not None:
                    cg_mod, cg_top = cg_attr
                    return (f"{a}.{b}", cg_mod, cg_top)
            return (f"{a}.{b}", src_module, src_symbol)

        if kind == "super_method":
            ## 1.0.5 P2: resolve super().method() owner from base classes.
            class_key, method = a, b
            tracer = tracers.get(module)
            if not tracer:
                return None
            bases = tracer.class_bases.get(class_key, [])
            resolved_owners = []
            for base_symbol in bases:
                if base_symbol in tracer.class_methods:
                    if method in tracer.class_methods[base_symbol]:
                        resolved_owners.append("local")
                        continue
                base_direct = normalize_source(tracer.symbols.direct.get(base_symbol))
                if isinstance(base_direct, CallResult):
                    base_direct = base_direct.callee
                if isinstance(base_direct, str) and base_direct not in ("local", "python", "unknown", ""):
                    top = self._top_source(
                        module, base_direct, tracers, _seen=set(_seen))
                    if top and top not in ("local", "python", "unknown", ""):
                        resolved_owners.append(top)
                        continue
                if isinstance(base_symbol, str) and '.' in base_symbol:
                    top = self._top_source(
                        module, base_symbol, tracers, _seen=set(_seen))
                    if top and top not in ("local", "python", "unknown", ""):
                        resolved_owners.append(top)
                        continue
            unique = list(dict.fromkeys(resolved_owners))  # preserve order
            if len(unique) == 1:
                return (f"super().{method}", module, unique[0])
            if len(unique) > 1:
                return (f"super().{method}", module, "unknown")
            return (f"super().{method}", module, "local")

        if kind == "container_iter":
            resolved = self._resolve_container_iter(module, a, tracers)
            if not resolved:
                ## Cannot determine element type — conservative fallback.
                return (f"{a}[*]", module, "unknown")
            src_module, candidates = resolved
            if len(candidates) == 1:
                src_symbol = candidates[0]
            elif any(candidate in ("local", "python", "unknown", "")
                     for candidate in candidates):
                src_symbol = "unknown"
            else:
                src_symbol = "[" + ",".join(candidates) + "]"
            return (f"{a}[*]", src_module, src_symbol)

        if kind == "call_result":
            callee = a
            # A call can be made through a structured receiver source, such
            # as a DataFrame column selected with ``df["col"]``. Resolve that
            # source before ordinary symbol tracing so reassignment preserves
            # the receiver owner instead of collapsing to a local callable.
            if isinstance(callee, ContainerItem):
                resolved_callee = self._resolve_structured_source(
                    module, callee, tracers, _seen=set(_seen))
                if resolved_callee is not None:
                    _, callee_module, callee_symbol = resolved_callee
                    callee_top = self._top_source(
                        callee_module, callee_symbol, tracers,
                        _seen=set(_seen))
                    if callee_top:
                        return (
                            f"{callee_display or source_display(callee)}()",
                            callee_module,
                            callee_top,
                        )
            ## 1.0.5 P2: explicit result_source carries result-object ownership.
            rs_explicit = getattr(direct_source, 'result_source', None)
            if rs_explicit is not None:
                if isinstance(rs_explicit, UnknownSource):
                    return (f"{callee_display or callee}()", module, "unknown")
                if isinstance(rs_explicit, PythonShape):
                    return (f"{callee_display or callee}()", module, "python")
                if rs_explicit == "python":
                    return (f"{callee_display or callee}()", module, "python")
                if isinstance(rs_explicit, DerivedResult):
                    ## 1.0.5 P2: resolve derived result from operands.
                    if rs_explicit.kind == "iterator":
                        return (f"{callee_display or callee}()",
                                module, "unknown")
                    candidates = self._origin_candidates(
                        module, rs_explicit, tracers,
                        _seen=set(_seen))
                    unique = self._dedupe_list([
                        candidate for candidate in candidates
                        if candidate not in ("", None)
                    ])
                    if len(unique) == 1 and unique[0] != "unknown":
                        return (f"{callee_display or callee}()",
                                module, unique[0])
                    # The placeholder may describe a method on a forwarded
                    # parameter.  Its exact local method summary can still
                    # prove the returned object at this call position.
                    bounded = self._bounded_call_result_candidates(
                        module, direct_source, tracers,
                        _seen=set(_seen))
                    if bounded is not None:
                        bounded_top = self._bounded_candidates_top(bounded)
                        bounded_module = self._bounded_owner_module(
                            bounded_top, module, tracers)
                        return (
                            f"{callee_display or callee}()",
                            bounded_module,
                            bounded_top,
                        )
                    return (f"{callee_display or callee}()", module, "unknown")
                elif isinstance(rs_explicit, str) and rs_explicit not in ("local", "unknown", ""):
                    # Module name string (from __import__("literal")) or
                    # other explicit library name.  This IS the top_library.
                    return (f"{callee_display or callee}()", module, rs_explicit)
            ## A method on an unconstrained function parameter does not prove
            ## the owner of its returned object.  Keep chained calls unknown
            ## unless an explicit result source above already resolved it.
            callee_source = normalize_source(callee)
            if (isinstance(callee_source, InstanceMethod)
                    and callee_source.parameter_scope):
                return (f"{callee_display or callee}()", module, "unknown")
            local_classes = self._local_class_from_method_result(
                module, direct_source, tracers)
            if len(local_classes) == 1:
                return (
                    f"{callee_display or callee}()",
                    local_classes[0][0],
                    local_classes[0][1],
                )
            if len(local_classes) > 1:
                return (f"{callee_display or callee}()", module, "unknown")
            bounded = self._bounded_call_result_candidates(
                module, direct_source, tracers, _seen=set(_seen))
            if bounded is not None:
                bounded_top = self._bounded_candidates_top(bounded)
                bounded_module = self._bounded_owner_module(
                    bounded_top, module, tracers)
                return (
                    f"{callee_display or callee}()",
                    bounded_module,
                    bounded_top,
                )
            cr_lineno = getattr(direct_source, 'call_lineno', 0) or 0
            cr_col = getattr(direct_source, 'call_col_offset', 0) or 0
            if isinstance(callee, SourceSet):
                primary = self._resolve_sourceset_primary(module, callee, tracers)
                if primary:
                    return (f"{callee_display or callee}()", module, primary)
                return (f"{callee_display or callee}()", module, "local")
            gs = getattr(self, '_call_searched_global', None)
            if gs is not None:
                if (module, callee) in gs:
                    callee_chain = [callee]
                else:
                    gs.add((module, callee))
                    callee_chain = self.trace_symbol(
                        module, callee, tracers, set(_seen))
            else:
                callee_chain = self.trace_symbol(
                    module, callee, tracers, set(_seen))
            def_module = module
            for item in reversed(callee_chain):
                if isinstance(item, str) and self.is_local(item):
                    def_module = item
                    break
            cur_module = def_module
            cur_symbol = callee
            # A direct import-from call preserves its qualified local symbol,
            # for example factory.create_app or provider.DecoderHolder.
            # Split the longest project-module prefix before looking up local
            # return summaries; external qualified names remain untouched.
            if (isinstance(callee, str)
                    and "." in callee
                    and callee not in tracers):
                parts = callee.split(".")
                for index in range(len(parts) - 1, 0, -1):
                    candidate_module = ".".join(parts[:index])
                    if candidate_module in tracers:
                        cur_module = candidate_module
                        cur_symbol = ".".join(parts[index:])
                        break
            seen = {(cur_module, cur_symbol)}
            while True:
                tr = tracers.get(cur_module)
                rs = tr.return_sources.get(cur_symbol) if tr else None
                # 1.0.5 P1: if callee is a module alias (import factory
                # as f; f.create_app()), resolve the function name from
                # display_name and look up return_sources in the target
                # module (via symbols.direct).
                if rs is None and tr is not None:
                    display_name = callee_display or ''
                    cur_is_simple = (not isinstance(cur_symbol, str)
                                     or '.' not in cur_symbol)
                    if '.' in display_name and cur_is_simple:
                        func_from_display = display_name.rsplit('.', 1)[-1]
                        rs = tr.return_sources.get(func_from_display)
                        # If not found, try the resolved module from
                        # symbols.direct (e.g. f → factory).
                        if rs is None:
                            mod_tracer = tracers.get(module)
                            if mod_tracer is not None:
                                first_seg = display_name.split('.')[0]
                                sd = mod_tracer.symbols.direct.get(first_seg)
                                if isinstance(sd, str) and sd in tracers:
                                    cur_module = sd
                                    tr = tracers[cur_module]
                                    rs = tr.return_sources.get(func_from_display)
                rs = normalize_source(rs)
                if isinstance(rs, SourceSet):
                    return_candidates = self._dedupe_list([
                        candidate for candidate in self._origin_candidates(
                            cur_module, rs, tracers, _seen=set(_seen))
                        if candidate not in ("", None)
                    ])
                    if return_candidates == ["python"]:
                        return (f"{callee_display or callee}()",
                                cur_module, "python")
                    concrete_returns = [
                        candidate for candidate in return_candidates
                        if candidate != "unknown"
                    ]
                    if ("python" in concrete_returns
                            and any(candidate not in ("python", "local")
                                    for candidate in concrete_returns)):
                        return (f"{callee_display or callee}()",
                                cur_module, "unknown")
                    ## 7B-full PR7-final: check primary convergence first
                    ## so that "return" origin can pick import-backed library even
                    ## when local sources are present.
                    primary = self._resolve_sourceset_primary(
                        cur_module, rs, tracers)
                    if primary:
                        return (f"{callee_display or callee}()",
                                cur_module, primary)
                    for src in rs.sources:
                        if isinstance(src, str):
                            arg_src = self._resolve_param_to_arg(
                                cur_module, cur_symbol, src, tracers,
                                call_lineno=cr_lineno, call_col_offset=cr_col)
                            if arg_src is not None:
                                arg_src = normalize_source(arg_src)
                                if isinstance(arg_src, CallResult):
                                    return (f"{callee_display or callee}()",
                                            cur_module, arg_src.callee)
                                if isinstance(arg_src, str):
                                    return (f"{callee_display or callee}()",
                                            cur_module, arg_src)
                            return (f"{callee_display or callee}()", cur_module, src)
                        if isinstance(src, CallResult):
                            return (f"{callee_display or callee}()", cur_module, src.callee)
                if rs is None:
                    ## 7B-full PR2: try call-graph return source before giving up.
                    cg_ret = self._lookup_cg_return_source(cur_module, cur_symbol)
                    if cg_ret is not None:
                        return (f"{callee_display or callee}()", module, cg_ret)
                    return (f"{callee_display or callee}()", cur_module, cur_symbol)
                if isinstance(rs, str):
                    arg_src = self._resolve_param_to_arg(
                        cur_module, cur_symbol, rs, tracers,
                        call_lineno=cr_lineno, call_col_offset=cr_col)
                    if arg_src is not None:
                        arg_src = normalize_source(arg_src)
                        if isinstance(arg_src, CallResult):
                            return (f"{callee_display or callee}()",
                                    cur_module, arg_src.callee)
                        if isinstance(arg_src, str):
                            return (f"{callee_display or callee}()",
                                    cur_module, arg_src)
                    return (f"{callee_display or callee}()", cur_module, rs)
                rs = normalize_source(rs)
                if isinstance(rs, CallResult):
                    next_chain = self.trace_symbol(cur_module, rs.callee, tracers, set())
                    cur_symbol = rs.callee
                    for item in reversed(next_chain):
                        if isinstance(item, str) and self.is_local(item):
                            cur_module = item
                            break
                    if (cur_module, cur_symbol) in seen:
                        return (f"{callee_display or callee}()", cur_module, cur_symbol)
                    seen.add((cur_module, cur_symbol))
                    continue
                break
            return (f"{callee_display or callee}()", cur_module, cur_symbol)

        return None

    ## Resolve a method receiver from all known parameter evidence.
    #
    #  A function parameter is not evidence of project-local ownership. A
    #  unique owner is returned only when call sites, static callbacks, or
    #  parameterization values converge. Uncalled and conflicting parameters
    #  remain unknown.
    #
    #  @param module Module containing the function parameter.
    #  @param method_source Parameter-backed InstanceMethod source.
    #  @param tracer Single-file analyzer for module.
    #  @param tracers Dict of module name to analyzer.
    #  @param _seen Structured-source recursion guard.
    #  @return Converged owner string or "unknown".
    def _resolve_parameter_method_top(self, module, method_source,
                                      tracer, tracers, _seen=None):
        parameter = method_source.parameter_name
        scope_name = method_source.parameter_scope
        if not parameter or not scope_name:
            return "unknown"

        receiver = method_source.receiver
        if (not isinstance(receiver, str)
                or not (receiver == parameter
                        or receiver.startswith(parameter + "."))):
            return "unknown"

        params = tracer.function_params.get(scope_name)
        if params is None:
            params = tracer.function_params.get(
                scope_name.rsplit(".", 1)[-1], [])
        if parameter not in params:
            return "unknown"

        param_index = params.index(parameter)
        call_arguments = self._parameter_call_arguments(
            module, scope_name, parameter, param_index, tracer, tracers,
            prefer_protocol_shape=True)
        if not call_arguments:
            return "unknown"

        receiver_class_filter = None
        scope_parts = scope_name.rsplit(".", 1)
        module_cg = getattr(self, "project_cg", None)
        defining_cg = (
            module_cg.modules.get(module)
            if module_cg is not None else None)
        if (len(scope_parts) == 2 and defining_cg is not None
                and scope_parts[0] in defining_cg.classes):
            receiver_class_filter = (module, scope_parts[0])

        attribute_path = (
            receiver[len(parameter) + 1:].split(".")
            if receiver != parameter else [])
        owners = []
        if not attribute_path:
            for arg_module, arg_source in call_arguments:
                if arg_source is None:
                    return "unknown"
                owners.extend(self._argument_method_owner_candidates(
                    arg_module, arg_source, method_source.method, tracers,
                    _seen=set(_seen or set()),
                    receiver_class_filter=receiver_class_filter))
        else:
            for arg_module, arg_source in call_arguments:
                if arg_source is None:
                    return "unknown"
                candidates = self._parameter_attribute_owner_candidates(
                    arg_module, arg_source, attribute_path, tracers)
                owners.extend(candidates)

        unique = self._dedupe_list([
            owner for owner in owners if owner not in (None, "")])
        if not unique or "unknown" in unique:
            return "unknown"
        if len(unique) == 1:
            return unique[0]
        return "unknown"

    ## Check whether an expression is composed only of direct parameters.
    #
    #  Subscript-derived parameters and local attributes are deliberately
    #  excluded. Their runtime element or attribute type is not established
    #  by the expression shape alone.
    #  @param source Source to inspect.
    #  @return True for a direct-parameter expression.
    def _is_direct_parameter_expression(self, source):
        source = normalize_source(source)
        if isinstance(source, ParameterSource):
            return not source.derived and not source.attributes
        if isinstance(source, DerivedResult):
            return (
                source.kind == "expression"
                and bool(source.sources)
                and all(self._is_direct_parameter_expression(item)
                        for item in source.sources)
            )
        return False

    ## Check whether every expression operand has bounded call-edge evidence.
    #  @param source Source or expression source to inspect.
    #  @return True for direct parameters and deferred instance fields.
    def _is_bounded_expression_source(self, source):
        source = normalize_source(source)
        if isinstance(source, InstanceAttribute):
            return True
        if isinstance(source, ParameterSource):
            return not source.derived and not source.attributes
        if isinstance(source, DerivedResult):
            return (
                source.kind == "expression"
                and bool(source.sources)
                and all(self._is_bounded_expression_source(item)
                        for item in source.sources)
            )
        return False

    ## Resolve a method on an expression derived from one or more parameters.
    #
    #  For example, ``combined = left + right`` followed by
    #  ``combined.reshape(...)`` carries two parameter sources.  Resolve each
    #  operand through its exact project call sites and accept the method
    #  owner only when every operand converges to the same owner.  This is a
    #  data-flow rule, not a library or method-name whitelist.
    #
    #  @param module Module containing the expression.
    #  @param source DerivedResult describing the expression operands.
    #  @param method Receiver method name.
    #  @param tracer Analyzer for the defining module.
    #  @param tracers Dict of module name to analyzer.
    #  @return One owner string, or "unknown" when the operands do not
    #  converge.
    def _resolve_derived_expression_method_top(self, module, source, method,
                                                tracer, tracers):
        external = self._resolve_expression_external_top(module, source, tracers)
        if external != "unknown":
            return external
        owners = []
        seen = set()

        def collect(value):
            value = normalize_source(value)
            if isinstance(value, DerivedResult):
                if value.kind != "expression" or not value.sources:
                    owners.append("unknown")
                    return
                for operand in value.sources:
                    collect(operand)
                return
            if isinstance(value, SourceSet):
                if not value.sources:
                    owners.append("unknown")
                    return
                for operand in value.sources:
                    collect(operand)
                return
            if isinstance(value, ParameterSource):
                key = (value.scope, value.name, method)
                if key in seen:
                    owners.append("unknown")
                    return
                seen.add(key)
                params = tracer.function_params.get(value.scope)
                if params is None:
                    params = tracer.function_params.get(
                        value.scope.rsplit(".", 1)[-1], [])
                if value.name not in params:
                    owners.append("unknown")
                    return
                arguments = self._parameter_call_arguments(
                    module, value.scope, value.name, params.index(value.name),
                    tracer, tracers, prefer_protocol_shape=True)
                if not arguments:
                    owners.append("unknown")
                    return
                for arg_module, arg_source in arguments:
                    candidates = self._argument_method_owner_candidates(
                        arg_module, arg_source, method, tracers)
                    owners.extend(candidates or ["unknown"])
                return
            candidates = self._argument_method_owner_candidates(
                module, value, method, tracers)
            owners.extend(candidates or ["unknown"])

        collect(source)
        unique = self._dedupe_list(
            owner for owner in owners if owner not in (None, ""))
        if len(unique) == 1 and unique[0] != "unknown":
            return unique[0]
        return "unknown"

    ## Resolve an expression receiver from converged import-backed operands.
    #
    #  This handles expressions that combine direct parameters with an
    #  independently resolved import-backed result.  The ordinary origin
    #  resolver follows parameter call edges and requires every operand to
    #  converge.  Python, local, unresolved, and conflicting candidates are
    #  deliberately rejected here because they do not prove one external
    #  receiver owner.
    #  @param module Module containing the expression.
    #  @param source DerivedResult describing the expression operands.
    #  @param tracers Dict of module name to analyzer.
    #  @return One import-backed owner string, or "unknown".
    def _resolve_expression_external_top(self, module, source, tracers):
        source = normalize_source(source)
        if (not isinstance(source, DerivedResult)
                or source.kind != "expression"
                or not source.sources):
            return "unknown"
        candidates = self._dedupe_list(
            self._origin_candidates(module, source, tracers))
        if (len(candidates) == 1
                and candidates[0] not in (
                    None, "", "local", "python", "unknown")):
            return candidates[0]
        return "unknown"

    ## Check whether an expression retains independent owner evidence.
    #
    #  When such evidence conflicts with parameter flow, the receiver must
    #  remain unknown instead of falling back to local.  Expressions made
    #  only from local and parameter sources retain the existing local
    #  identity contract.
    #  @param source Expression source to inspect.
    #  @return True when an operand carries non-local owner evidence.
    def _has_expression_owner_evidence(self, source):
        source = normalize_source(source)
        if isinstance(source, (CallResult, PythonShape)):
            return True
        if isinstance(source, SourceSet):
            return any(self._has_expression_owner_evidence(item)
                       for item in source.sources)
        if isinstance(source, DerivedResult):
            return any(self._has_expression_owner_evidence(item)
                       for item in source.sources)
        if isinstance(source, str):
            return source not in ("", "local", "python", "unknown")
        return False

    ## Resolve argument ownership for one concrete receiver method.
    #
    #  PythonShape values must support the requested builtin protocol.
    #  Forwarded parameters are followed recursively; all other sources use
    #  ordinary owner resolution.
    #  @param module Module containing the argument expression.
    #  @param source Argument source.
    #  @param method Receiver method name.
    #  @param tracers Dict of module name to analyzer.
    #  @param _seen Parameter recursion guard.
    #  @param receiver_class_filter Optional project-local virtual-dispatch
    #  class context retained while following forwarded parameters.
    #  @param _context Exact local return context, before owner projection.
    #  @return Candidate owner strings.
    def _argument_method_owner_candidates(
            self, module, source, method, tracers, _seen=None,
            receiver_class_filter=None, _context=None):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            candidates = []
            for item in source.sources:
                candidates.extend(self._argument_method_owner_candidates(
                    module, item, method, tracers, _seen,
                    receiver_class_filter, _context))
            return self._dedupe_list(candidates) or ["unknown"]
        if isinstance(source, CallResult):
            shape = self._returned_python_shape(
                module, source, tracers, _context)
            if shape is not None:
                return self._argument_method_owner_candidates(
                    module, shape, method, tracers, _seen,
                    receiver_class_filter, _context)
            candidates = self._bounded_call_result_method_candidates(
                module, source, method, tracers, _context, _seen,
                receiver_class_filter)
            if candidates is not None:
                return candidates
            if isinstance(source.result_source, (PythonShape, SourceSet, CallResult,
                                                 InstanceMethod, DerivedResult)):
                return self._argument_method_owner_candidates(
                    source.source_module or module, source.result_source,
                    method, tracers, _seen, receiver_class_filter, _context)
        if isinstance(source, ParameterSource) and source.derived:
            shape = self._returned_python_shape(module, source, tracers, _context)
            if shape is not None:
                return self._argument_method_owner_candidates(
                    module, shape, method, tracers, _seen,
                    receiver_class_filter, _context)
        if isinstance(source, ParameterSource) and _context is not None:
            current = _context
            while current is not None:
                if (module == current.target.module
                        and source.scope == current.target.qualname):
                    if source.attributes or source.derived:
                        # Preserve the established import-backed surface
                        # contract. A derived Python value still needs an
                        # explicit item/attribute shape, not just its owner.
                        owners = self._bounded_source_candidates(
                            module, source, _context, tracers,
                            set(_seen or set()))
                        if owners and all(owner not in (
                                None, "", "local", "python", "unknown")
                                for owner in owners):
                            return owners
                        return ["unknown"]
                    summary = self.project_cg.modules[
                        module].functions[current.target.qualname]
                    if source.name not in summary.params:
                        return ["unknown"]
                    arguments = self._edge_parameter_sources(
                        current.edge, summary, source.name,
                        summary.params.index(source.name),
                        prefer_protocol_shape=True)
                    candidates = []
                    for argument in arguments or [UnknownSource()]:
                        candidates.extend(self._argument_method_owner_candidates(
                            current.caller_module, argument, method, tracers,
                            _seen, receiver_class_filter, current.parent))
                    return self._dedupe_list(candidates) or ["unknown"]
                current = current.parent
        if isinstance(source, (ContainerIter, ContainerItem)):
            _seen = set(_seen or set())
            key = ("container-method", module, type(source).__name__,
                   source_display(source), method, receiver_class_filter)
            if key in _seen:
                return ["unknown"]
            _seen.add(key)
        if isinstance(source, InstanceAttribute):
            return [self._resolve_instance_attribute_method_top(
                module, source, method, tracers,
                _seen=set(_seen or set()))]
        if isinstance(source, PythonShape):
            return (["python"] if _has_builtin_shape_method(source.kind, method)
                    else ["unknown"])
        if isinstance(source, InstanceMethod):
            result_shape = self._returned_python_shape(module, source, tracers, _context)
            if result_shape is not None:
                return self._argument_method_owner_candidates(
                    module, result_shape, method, tracers, _seen,
                    receiver_class_filter, _context)
            receiver = source.receiver
            if source.parameter_scope and receiver == source.parameter_name:
                receiver = ParameterSource(source.parameter_scope, source.parameter_name)
            if self._returned_python_shape(module, receiver, tracers, _context) is not None:
                return ["unknown"]
        if isinstance(source, DerivedResult):
            if source.kind == "tuple":
                return self._argument_method_owner_candidates(
                    module, PythonShape("tuple"), method, tracers, _seen,
                    receiver_class_filter, _context)
            if self._is_bounded_expression_source(source):
                candidates = []
                for operand in source.sources:
                    candidates.extend(
                        self._argument_method_owner_candidates(
                            module, operand, method, tracers, _seen,
                            receiver_class_filter=receiver_class_filter,
                            _context=_context))
                unique = self._dedupe_list(candidates)
                if len(unique) == 1 and unique[0] != "unknown":
                    return unique
            external = self._resolve_expression_external_top(
                module, source, tracers)
            if external != "unknown":
                return [external]
            return ["unknown"]
        if isinstance(source, ContainerIter):
            elements = self._returned_element_sources(
                module, source.container, tracers, iterable=True,
                _seen=_seen)
            if elements is not None:
                candidates = []
                for origin, element in elements:
                    candidates.extend(self._argument_method_owner_candidates(
                        origin, element, method, tracers, _seen,
                        receiver_class_filter=receiver_class_filter,
                        _context=_context))
                return self._dedupe_list(candidates) or ["unknown"]
            resolved = self._resolve_container_iter(
                module, source.container, tracers)
            if resolved is None:
                return ["unknown"]
            source_module, element_sources = resolved
            candidates = []
            for element_source in element_sources:
                candidates.extend(self._argument_method_owner_candidates(
                    source_module, element_source, method, tracers,
                    _seen, receiver_class_filter=receiver_class_filter,
                    _context=_context))
            return self._dedupe_list(candidates) or ["unknown"]
        if isinstance(source, ContainerItem):
            container = normalize_source(source.container)
            if isinstance(container, PythonShape):
                item_kind = container.item_kind
                if container.kind in ("str", "bytes"):
                    item_kind = "str" if container.kind == "str" else "int"
                if item_kind:
                    return self._argument_method_owner_candidates(
                        module, PythonShape(item_kind), method, tracers,
                        _seen, receiver_class_filter=receiver_class_filter,
                        _context=_context)
                return ["unknown"]
            selected = _tuple_source_item(container, source.index)
            if selected is not None:
                return self._argument_method_owner_candidates(
                    module, selected, method, tracers, _seen,
                    receiver_class_filter=receiver_class_filter,
                    _context=_context)
            if isinstance(container, CallResult):
                candidates = self._bounded_call_result_item_candidates(
                    module, container, source.index, tracers, _seen=_seen)
                if candidates is not None:
                    return candidates
            if isinstance(container, ParameterSource):
                seen = set(_seen or set())
                key = ("item-method", module, container.scope,
                       container.name, source.index, method)
                if key in seen:
                    return ["unknown"]
                seen.add(key)
                tracer = tracers.get(module)
                if tracer is None:
                    return ["unknown"]
                params = tracer.function_params.get(container.scope)
                if params is None:
                    params = tracer.function_params.get(
                        container.scope.rsplit(".", 1)[-1], [])
                if container.name in params:
                    target_module_cg = self.project_cg.modules.get(module)
                    summary = (target_module_cg.functions.get(container.scope)
                               if target_module_cg is not None else None)
                    is_variadic = (
                        summary is not None
                        and container.name in (
                            getattr(summary, "vararg", ""),
                            getattr(summary, "kwarg", "")))
                    if not is_variadic:
                        arguments = self._parameter_call_arguments(
                            module, container.scope, container.name,
                            params.index(container.name), tracer, tracers,
                            prefer_protocol_shape=True)
                        if not arguments:
                            return ["unknown"]
                        candidates = []
                        for arg_module, arg_source in arguments:
                            selected = _tuple_source_item(
                                arg_source, source.index)
                            if selected is None:
                                selected = ContainerItem(
                                    arg_source, source.index)
                            candidates.extend(
                                self._argument_method_owner_candidates(
                                    arg_module, selected, method, tracers,
                                    seen, receiver_class_filter=
                                    receiver_class_filter, _context=_context))
                        return self._dedupe_list(candidates) or ["unknown"]
                    arguments = self._parameter_pack_item_arguments(
                        module, container, source.index, tracers)
                    if not arguments:
                        return ["unknown"]
                    candidates = []
                    for arg_module, arg_source in arguments:
                        candidates.extend(
                            self._argument_method_owner_candidates(
                                arg_module, arg_source, method, tracers,
                                seen, receiver_class_filter=receiver_class_filter,
                                _context=_context))
                    return self._dedupe_list(candidates) or ["unknown"]
                return ["unknown"]
            # Selecting an item does not inherit the aggregate object's
            # owner.  Follow only an item source explicitly recorded from
            # project syntax; imported call results and opaque containers
            # otherwise remain unknown.
            resolved = self._resolve_container_item(
                module, source.container, source.index, tracers)
            if resolved is None:
                return ["unknown"]
            item_module, item_source = resolved
            return self._argument_method_owner_candidates(
                item_module, item_source, method, tracers, _seen,
                receiver_class_filter=receiver_class_filter,
                _context=_context)
        if not isinstance(source, ParameterSource):
            return self._origin_candidates(
                module, source, tracers,
                _seen=set(_seen or set()))
        if source.attributes:
            return ["unknown"]
        if (source.derived
                and source.derived_operation != "slice"):
            return ["unknown"]

        seen = set(_seen or set())
        key = (
            module, source.scope, source.name, method,
            source.derived_operation, receiver_class_filter,
        )
        if key in seen:
            return []
        seen.add(key)
        tracer = tracers.get(module)
        if tracer is None:
            return ["unknown"]
        params = tracer.function_params.get(source.scope)
        if params is None:
            params = tracer.function_params.get(
                source.scope.rsplit(".", 1)[-1], [])
        if source.name not in params:
            return ["unknown"]
        arguments = self._parameter_call_arguments(
            module, source.scope, source.name,
            params.index(source.name), tracer, tracers,
            prefer_protocol_shape=True,
            receiver_class_filter=receiver_class_filter)
        if not arguments:
            return ["unknown"]
        candidates = []
        for arg_module, arg_source in arguments:
            if source.derived:
                arg_source = normalize_source(arg_source)
                if isinstance(arg_source, PythonShape):
                    if arg_source.kind not in (
                            "str", "bytes", "list", "tuple"):
                        candidates.append("unknown")
                        continue
                elif not isinstance(arg_source, ParameterSource):
                    candidates.append("unknown")
                    continue
            candidates.extend(self._argument_method_owner_candidates(
                arg_module, arg_source, method, tracers, seen,
                receiver_class_filter=receiver_class_filter,
                _context=_context))
        return self._dedupe_list(candidates)

    ## Validate a local call result's method before projecting return owners.
    #  @param module Module containing the result-producing call.
    #  @param source CallResult with the exact call-site position.
    #  @param method Requested receiver method.
    #  @param tracers Per-module analyzers.
    #  @param parent Enclosing forwarding context.
    #  @param seen Recursion guard.
    #  @param receiver_class_filter Optional local receiver class context.
    #  @return Candidate owners, or None when no local call context exists.
    def _bounded_call_result_method_candidates(
            self, module, source, method, tracers, parent, seen,
            receiver_class_filter):
        module = source.source_module or module
        contexts = self._bounded_call_contexts(
            module, source.call_lineno, source.call_col_offset,
            tracers, parent=parent, callee_name=source.display_name)
        if not contexts:
            return None
        # A constructor edge targets __init__, whose return is not the
        # instance returned by the class call. Keep class-owner resolution.
        if any(context.target.qualname.endswith(".__init__")
               and context.edge.callee_name.rsplit(".", 1)[-1] != "__init__"
               for context in contexts):
            return None
        candidates = []
        for context in contexts:
            key = ("return-method", module, source.call_lineno,
                   source.call_col_offset, context.target, method)
            if key in (seen or set()):
                candidates.append("unknown")
                continue
            context_seen = set(seen or set())
            context_seen.add(key)
            module_cg = self.project_cg.modules.get(context.target.module)
            summary = (module_cg.functions.get(context.target.qualname)
                       if module_cg is not None else None)
            return_values = (summary.return_values if summary is not None
                             else None)
            if return_values is None and summary is not None:
                return_values = summary.returns
            if return_values is None:
                candidates.append("unknown")
                continue
            if (len(context.edge.assigned_to) > 1
                    and self._is_tuple_return_source(return_values)):
                # Some legacy unpacked arguments retain the producing call
                # but not the selected index. Accept only a protocol shared
                # by every possible item, never the aggregate tuple owner.
                branches = (return_values.sources
                            if isinstance(return_values, SourceSet)
                            else (return_values,))
                return_values = make_source_set(
                    [item for branch in branches for item in branch.sources],
                    origin="return")
            candidates.extend(self._argument_method_owner_candidates(
                context.target.module, return_values, method, tracers,
                context_seen, receiver_class_filter, context))
        return self._dedupe_list(candidates) or ["unknown"]

    ## Resolve owner candidates for an argument, following parameter forwarding.
    #  @param module Module containing the argument expression.
    #  @param source Argument source.
    #  @param tracers Dict of module name to analyzer.
    #  @param _seen Parameter recursion guard.
    #  @return Candidate owner strings.
    def _argument_owner_candidates(self, module, source, tracers, _seen=None):
        source = normalize_source(source)
        if source == "local":
            return ["local"]
        if isinstance(source, ContainerIter):
            container = normalize_source(source.container)
            if isinstance(container, ParameterSource):
                # An iterable element from a parameter is not evidence of a
                # project-local receiver.  Returning unknown here also keeps
                # local call-target matching from recursively resolving the
                # same parameterization edge.
                return ["unknown"]
        if isinstance(source, ContainerItem):
            container = normalize_source(source.container)
            if isinstance(container, ParameterSource):
                seen = set(_seen or set())
                key = ("pack-item", module, container.scope,
                       container.name, source.index)
                if key in seen:
                    return ["unknown"]
                seen.add(key)
                tracer = tracers.get(module)
                if tracer is None:
                    return ["unknown"]
                params = tracer.function_params.get(container.scope)
                if params is None:
                    params = tracer.function_params.get(
                        container.scope.rsplit(".", 1)[-1], [])
                if container.name in params:
                    arguments = self._parameter_pack_item_arguments(
                        module, container, source.index, tracers)
                    if not arguments:
                        return ["unknown"]
                    candidates = []
                    for arg_module, arg_source in arguments:
                        candidates.extend(self._argument_owner_candidates(
                            arg_module, arg_source, tracers, seen))
                    return self._dedupe_list(candidates) or ["unknown"]
                return ["unknown"]
        if not isinstance(source, ParameterSource):
            return self._origin_candidates(module, source, tracers)
        if source.derived or source.attributes:
            return ["unknown"]

        seen = set(_seen or set())
        key = (module, source.scope, source.name)
        if key in seen:
            return []
        seen.add(key)
        tracer = tracers.get(module)
        if tracer is None:
            return ["unknown"]
        params = tracer.function_params.get(source.scope)
        if params is None:
            params = tracer.function_params.get(
                source.scope.rsplit(".", 1)[-1], [])
        if source.name not in params:
            return ["unknown"]
        arguments = self._parameter_call_arguments(
            module, source.scope, source.name,
            params.index(source.name), tracer, tracers)
        if not arguments:
            return ["unknown"]
        candidates = []
        for arg_module, arg_source in arguments:
            candidates.extend(self._argument_owner_candidates(
                arg_module, arg_source, tracers, seen))
        return self._dedupe_list(candidates)

    ## Resolve a dotted parameter receiver through a local class attribute.
    #
    #  Supports bounded paths such as holder.payload.method() when holder is
    #  constructed locally and self.payload is bound from a constructor
    #  argument. Every constructor value must converge before ownership is
    #  returned.
    #  @param module Module containing the root argument.
    #  @param source Root argument source.
    #  @param attributes Receiver attributes after the parameter name.
    #  @param tracers Dict of module name to analyzer.
    #  @return Candidate owner strings.
    def _parameter_attribute_owner_candidates(self, module, source,
                                               attributes, tracers):
        if len(attributes) != 1:
            return ["unknown"]
        class_ref = self._local_class_from_source(module, source)
        if class_ref is None:
            return ["unknown"]
        class_module, class_name = class_ref
        module_cg = getattr(self, "project_cg", None)
        if module_cg is None or class_module not in module_cg.modules:
            return ["unknown"]
        class_summary = module_cg.modules[class_module].classes.get(class_name)
        class_tracer = tracers.get(class_module)
        if class_summary is None or class_tracer is None:
            return ["unknown"]
        attr_source = class_summary.attrs.get("self." + attributes[0])
        if attr_source is None:
            return ["unknown"]

        init_scope = class_name + ".__init__"
        init_params = class_tracer.function_params.get(init_scope, [])
        if isinstance(attr_source, str) and attr_source in init_params:
            attr_arguments = self._parameter_call_arguments(
                class_module, init_scope, attr_source,
                init_params.index(attr_source), class_tracer, tracers)
        else:
            attr_arguments = [(class_module, attr_source)]
        if not attr_arguments:
            return ["unknown"]
        candidates = []
        for arg_module, arg_source in attr_arguments:
            candidates.extend(self._argument_owner_candidates(
                arg_module, arg_source, tracers))
        return self._dedupe_list(candidates)

    ## Resolve a base-method field read through concrete subclass call edges.
    #
    #  A base class may read ``self.payload`` while a local subclass assigns
    #  that field from one of its method parameters. The lexical class alone
    #  cannot resolve the field. This helper finds the local runtime classes
    #  that actually reach the containing method, follows their field
    #  bindings through project call edges, and accepts only one converged
    #  callable owner.
    #  @param module Module containing the field read.
    #  @param source Structured instance-attribute source.
    #  @param method Method called on the field value.
    #  @param tracers Dict of module name to analyzer.
    #  @param _seen Structured-source recursion guard.
    #  @return Converged owner string or "unknown".
    def _resolve_instance_attribute_method_top(
            self, module, source, method, tracers, _seen=None):
        if not source.scope or not source.class_name or not source.attribute:
            return "unknown"

        base_identity = (module, source.class_name)
        runtime_classes = []
        for caller_module, caller_cg in self.project_cg.modules.items():
            caller_tracer = tracers.get(caller_module)
            for edge in caller_cg.edges:
                if not self._edge_targets_local_function(
                        edge, caller_module, module, source.scope,
                        caller_tracer, tracers,
                        allow_inherited_dispatch=True):
                    continue
                candidates = self._local_class_candidates(
                    caller_module, edge.receiver_source, tracers)
                if (not candidates and edge.receiver_source == "self"):
                    caller_parts = edge.caller.qualname.rsplit(".", 1)
                    if len(caller_parts) == 2:
                        caller_class = caller_parts[0]
                        module_cg = self.project_cg.modules.get(caller_module)
                        if (module_cg is not None
                                and caller_class in module_cg.classes):
                            candidates = [(caller_module, caller_class)]
                for candidate in candidates:
                    if self._local_class_is_or_derives(
                            candidate[0], candidate[1],
                            base_identity[0], base_identity[1], tracers):
                        runtime_classes.append(candidate)

        runtime_classes = self._dedupe_list(runtime_classes)
        if not runtime_classes:
            return "unknown"

        owners = []
        for runtime_module, runtime_class in runtime_classes:
            bindings = self._local_class_attribute_bindings(
                runtime_module, runtime_class, source.attribute, tracers)
            if not bindings:
                owners.append("unknown")
                continue
            for binding_module, binding_source in bindings:
                candidates = self._argument_method_owner_candidates(
                    binding_module, binding_source, method, tracers,
                    _seen=set(_seen or set()),
                    receiver_class_filter=(runtime_module, runtime_class))
                owners.extend(candidates or ["unknown"])

        unique = self._dedupe_list(
            owner for owner in owners if owner not in (None, ""))
        if len(unique) == 1 and unique[0] != "unknown":
            return unique[0]
        return "unknown"

    ## Find one instance-field binding on a local class or its local bases.
    #  @param module Defining module of the candidate class.
    #  @param class_name Candidate class name.
    #  @param attribute Normalized ``self.<field>`` path.
    #  @param tracers Dict of module name to analyzer.
    #  @param visited Inheritance recursion guard.
    #  @return List of (module, source) bindings.
    def _local_class_attribute_bindings(
            self, module, class_name, attribute, tracers, visited=None):
        identity = (module, class_name)
        seen = set(visited or set())
        if identity in seen:
            return []
        seen.add(identity)
        module_cg = self.project_cg.modules.get(module)
        class_summary = (
            module_cg.classes.get(class_name)
            if module_cg is not None else None)
        if class_summary is None:
            return []
        if attribute in class_summary.attrs:
            return [(module, class_summary.attrs[attribute])]

        bindings = []
        tracer = tracers.get(module)
        if tracer is None:
            return []
        for base_symbol in tracer.class_bases.get(class_name, []):
            base_identity = self._resolve_local_class_identity(
                module, base_symbol, tracers)
            if base_identity is None:
                continue
            bindings.extend(self._local_class_attribute_bindings(
                base_identity[0], base_identity[1], attribute, tracers,
                seen))
        return self._dedupe_list(bindings)

    ## Identify a project-local class constructor source.
    #  @param module Module containing the source.
    #  @param source Source to inspect.
    #  @return (module, class name) or None.
    def _local_class_from_source(self, module, source):
        source = normalize_source(source)
        if isinstance(source, CallResult):
            source = normalize_source(source.callee)
        if not isinstance(source, str):
            return None
        cg = getattr(self, "project_cg", None)
        if cg is None:
            return None
        if module in cg.modules and source in cg.modules[module].classes:
            return (module, source)
        parts = source.split(".")
        for index in range(len(parts) - 1, 0, -1):
            candidate_module = ".".join(parts[:index])
            candidate_class = ".".join(parts[index:])
            module_cg = cg.modules.get(candidate_module)
            if module_cg and candidate_class in module_cg.classes:
                return (candidate_module, candidate_class)
        return None

    ## Resolve a local class returned by a local class or static method.
    #
    #  This follows only an explicit return summary. It does not infer a
    #  class from a method name or from a variable spelling. Every return
    #  branch must resolve to a project-local class; multiple classes remain
    #  ambiguous and are resolved conservatively by the caller.
    #  @param module Module containing the call result.
    #  @param source CallResult representing a method call.
    #  @param tracers Dict of module name to analyzer.
    #  @return List of unique local class identities.
    def _local_class_from_method_result(self, module, source, tracers):
        source = normalize_source(source)
        if not isinstance(source, CallResult):
            return []
        callee = source.callee
        if not isinstance(callee, str) or "." not in callee:
            return []
        caller_tracer = tracers.get(module)
        identities = []
        for candidate_module, module_cg in self.project_cg.modules.items():
            for class_name, class_summary in module_cg.classes.items():
                for method_name, method_summary in class_summary.methods.items():
                    qualified_class = candidate_module + "." + class_name
                    qualified_method = qualified_class + "." + method_name
                    spellings = {qualified_method}
                    if candidate_module == module:
                        spellings.add(class_name + "." + method_name)
                    if caller_tracer is not None:
                        parts = callee.split(".")
                        prefix = parts[0]
                        imported = caller_tracer.import_from_symbols.get(
                            prefix)
                        if (isinstance(imported, str)
                                and imported + "." + method_name
                                == qualified_method):
                            spellings.add(callee)
                        direct = normalize_source(
                            caller_tracer.symbols.direct.get(prefix))
                        if (isinstance(direct, str)
                                and direct == candidate_module
                                and len(parts) == 3
                                and parts[1] == class_name
                                and parts[2] == method_name):
                            spellings.add(callee)
                    if callee not in spellings:
                        continue
                    returns = normalize_source(method_summary.returns)
                    if returns is None:
                        continue
                    return_sources = (
                        list(returns.sources)
                        if isinstance(returns, SourceSet)
                        else [returns]
                    )
                    returned_classes = []
                    for returned in return_sources:
                        returned = normalize_source(returned)
                        if returned == "self":
                            returned_classes.append(
                                (candidate_module, class_name))
                            continue
                        if isinstance(returned, CallResult):
                            returned = returned.callee
                        identity = self._local_class_from_source(
                            candidate_module, returned)
                        if identity is None:
                            returned_classes = []
                            break
                        returned_classes.append(identity)
                    if returned_classes:
                        identities.extend(self._dedupe_list(returned_classes))
        return self._dedupe_list(identities)

    ## Resolve local classes returned by a project-local function.
    #
    #  The function must have an explicit return summary whose every branch
    #  resolves to one project-local constructor.  This deliberately does not
    #  infer a class from a function name, argument type, or method spelling.
    #  @param module Module containing the call site.
    #  @param callee Callee spelling from a CallResult.
    #  @param tracers Dict of module name to analyzer.
    #  @return List of unique local class identities.
    def _local_classes_from_function_result(self, module, callee, tracers):
        if not isinstance(callee, str):
            return []
        candidates = []
        caller_tracer = tracers.get(module)
        if caller_tracer is not None:
            imported = caller_tracer.import_from_symbols.get(callee)
            if isinstance(imported, str):
                candidates.append((module, imported))

        parts = callee.split(".")
        for index in range(len(parts), 0, -1):
            candidate_module = ".".join(parts[:index])
            if candidate_module in tracers:
                candidates.append((
                    candidate_module, ".".join(parts[index:])))
                break
        candidates.append((module, callee))

        identities = []
        seen = set()
        for target_module, qualname in candidates:
            key = (target_module, qualname)
            if key in seen or not qualname:
                continue
            seen.add(key)
            tracer = tracers.get(target_module)
            if tracer is None:
                continue
            returns = tracer.return_sources.get(qualname)
            if returns is None and target_module == module:
                returns = tracer.return_sources.get(parts[-1])
            if returns is None:
                continue
            normalized = normalize_source(returns)
            returned_sources = (
                list(normalized.sources)
                if isinstance(normalized, SourceSet)
                else [normalized]
            )
            branch_identities = []
            for returned in returned_sources:
                returned = normalize_source(returned)
                if isinstance(returned, CallResult):
                    returned = returned.callee
                identity = self._local_class_from_source(
                    target_module, returned)
                if identity is None:
                    branch_identities = []
                    break
                branch_identities.append(identity)
            if branch_identities:
                identities.extend(branch_identities)
        return self._dedupe_list(identities)

    ## Check whether a local class owns a method through local inheritance.
    #  @param module Defining module of the class.
    #  @param class_name Local class name.
    #  @param method_name Method name.
    #  @param tracers Dict of module name to analyzer.
    #  @param visited Recursion guard.
    #  @return True when the method is locally defined or locally inherited.
    def _local_class_defines_method(self, module, class_name, method_name,
                                    tracers, visited=None):
        key = (module, class_name, method_name)
        seen = set(visited or set())
        if key in seen:
            return False
        seen.add(key)
        module_cg = self.project_cg.modules.get(module)
        if module_cg is None:
            return False
        class_summary = module_cg.classes.get(class_name)
        if class_summary is None:
            return False
        if method_name in class_summary.methods:
            return True
        tracer = tracers.get(module)
        if tracer is None:
            return False
        for base_symbol in class_summary.bases:
            base_identity = self._resolve_local_class_identity(
                module, base_symbol, tracers)
            if base_identity and self._local_class_defines_method(
                    base_identity[0], base_identity[1], method_name,
                    tracers, seen):
                return True
        return False

    ## Resolve a project-local class through imports and package re-exports.
    #
    #  @param module Module where the class symbol is referenced.
    #  @param source Class symbol or constructor source.
    #  @param tracers Dict of module name to analyzer.
    #  @return (defining module, class name) or None.
    def _resolve_local_class_identity(self, module, source, tracers):
        direct = self._local_class_from_source(module, source)
        if direct is not None:
            return direct
        source = normalize_source(source)
        if isinstance(source, CallResult):
            source = normalize_source(source.callee)
        if not isinstance(source, str):
            return None
        chain = self.trace_symbol(module, source, tracers, set())
        cg = getattr(self, "project_cg", None)
        if cg is None:
            return None
        for index in range(len(chain) - 1, 0, -1):
            class_name = chain[index]
            class_module = chain[index - 1]
            if not isinstance(class_name, str):
                continue
            module_cg = cg.modules.get(class_module)
            if module_cg and class_name in module_cg.classes:
                return (class_module, class_name)
        return None

    ## Return whether one project-local class derives from another.
    #
    #  @param candidate_module Defining module of the candidate subclass.
    #  @param candidate_class Candidate subclass name.
    #  @param base_module Defining module of the required base class.
    #  @param base_class Required base class name.
    #  @param tracers Dict of module name to analyzer.
    #  @param visited Recursion guard for cyclic or malformed hierarchies.
    #  @return True for identity or transitive local inheritance.
    def _local_class_is_or_derives(
            self, candidate_module, candidate_class,
            base_module, base_class, tracers, visited=None):
        candidate = (candidate_module, candidate_class)
        required = (base_module, base_class)
        if candidate == required:
            return True
        seen = set(visited or set())
        if candidate in seen:
            return False
        seen.add(candidate)
        tracer = tracers.get(candidate_module)
        if tracer is None:
            return False
        for base_symbol in tracer.class_bases.get(candidate_class, []):
            identity = self._resolve_local_class_identity(
                candidate_module, base_symbol, tracers)
            if identity is None:
                continue
            if self._local_class_is_or_derives(
                    identity[0], identity[1],
                    base_module, base_class, tracers, seen):
                return True
        return False

    ## Collect project-local runtime class candidates for a receiver source.
    #
    #  Explicit constructors and statically tracked container elements provide
    #  class-level dispatch evidence. An empty result means the source cannot
    #  constrain dispatch, not that the receiver is non-local.
    #  @param module Module where the source is evaluated.
    #  @param source Receiver source.
    #  @param tracers Dict of module name to analyzer.
    #  @param visited Recursion guard.
    #  @return List of (defining module, class name) tuples.
    def _local_class_candidates(
            self, module, source, tracers, visited=None):
        source = normalize_source(source)
        key = (module, type(source).__name__, source_display(source))
        seen = set(visited or set())
        if key in seen:
            return []
        seen.add(key)

        if isinstance(source, CallResult):
            method_classes = self._local_class_from_method_result(
                module, source, tracers)
            if method_classes:
                return method_classes
            function_classes = self._local_classes_from_function_result(
                module, source.callee, tracers)
            if function_classes:
                return function_classes
            callable_classes = self._local_callable_class_candidates(
                module, source, tracers.get(module), tracers, seen)
            if callable_classes:
                return callable_classes
            identity = self._local_class_from_source(
                module, source.callee)
            return [identity] if identity is not None else []
        identity = self._resolve_local_class_identity(
            module, source, tracers)
        if identity is not None:
            return [identity]
        if isinstance(source, SourceSet):
            candidates = []
            for item in source.sources:
                candidates.extend(self._local_class_candidates(
                    module, item, tracers, set(seen)))
            return self._dedupe_list(candidates)
        if isinstance(source, ContainerItem):
            resolved = self._resolve_container_item(
                module, source.container, source.index, tracers)
            if resolved is None:
                return []
            return self._local_class_candidates(
                resolved[0], resolved[1], tracers, seen)
        if isinstance(source, ContainerIter):
            container = normalize_source(source.container)
            if not isinstance(container, str):
                return self._local_class_candidates(
                    module, container, tracers, seen)
            tracer = tracers.get(module)
            if tracer is None:
                return []
            candidates = []
            for (container_name, _), item_source in (
                    tracer.container_items.items()):
                if container_name == container:
                    candidates.extend(self._local_class_candidates(
                        module, item_source, tracers, set(seen)))
            for item_source in tracer.container_set_sources.get(
                    container, set()):
                candidates.extend(self._local_class_candidates(
                    module, item_source, tracers, set(seen)))
            return self._dedupe_list(candidates)
        return []

    ## Resolve a constructor-injected callable field of a parameter object.
    #  @param module Module containing the field call.
    #  @param source Parameter-backed field-call source.
    #  @param tracers All project analyzers.
    #  @param seen Callable-source recursion guard.
    #  @return Complete local callable class candidates, or an empty list.
    def _parameter_callable_field_classes(self, module, source, tracers, seen):
        key = (module, source.parameter_scope, source.parameter_name, source.method)
        if key in self._callable_field_in_progress:
            return []
        tracer = tracers.get(module)
        params = (tracer.function_params.get(source.parameter_scope, [])
                  if tracer is not None else [])
        if source.parameter_name not in params:
            return []
        # A write outside an initializer invalidates constructor-only evidence.
        if self._constructor_only_fields is None:
            fields, blocked = set(), set()
            for candidate in tracers.values():
                tree = candidate._module_tree
                if tree is None:
                    continue
                parents = {id(child): node for node in ast.walk(tree)
                           for child in ast.iter_child_nodes(node)}
                for node in ast.walk(tree):
                    if (not isinstance(node, ast.Attribute)
                            or not isinstance(node.ctx, (ast.Store, ast.Del))):
                        continue
                    parent = parents.get(id(node))
                    while parent is not None and not isinstance(
                            parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        parent = parents.get(id(parent))
                    fields.add(node.attr)
                    if (not isinstance(node.ctx, ast.Store)
                            or not isinstance(node.value, ast.Name)
                            or node.value.id != 'self'
                            or parent is None or parent.name != '__init__'
                            or not isinstance(parents.get(id(parent)), ast.ClassDef)):
                        blocked.add(node.attr)
            self._constructor_only_fields = fields - blocked
        if source.method not in self._constructor_only_fields:
            return []
        self._callable_field_in_progress.add(key)
        try:
            arguments = self._parameter_call_arguments(
                module, source.parameter_scope, source.parameter_name,
                params.index(source.parameter_name), tracer, tracers)
            if self._callable_field_receiver_escapes(module, source, arguments, tracers):
                return []
            classes = []
            for origin, receiver in arguments:
                identity = self._resolve_local_class_identity(origin, receiver, tracers)
                if identity is None or not isinstance(receiver, CallResult):
                    return []
                bindings = self._local_class_attribute_bindings(
                    identity[0], identity[1], 'self.' + source.method, tracers)
                if not bindings:
                    return []
                contexts = self._bounded_call_contexts(
                    origin, receiver.call_lineno, receiver.call_col_offset, tracers,
                    callee_name=receiver.display_name)
                for binding_module, binding in bindings:
                    binding = normalize_source(binding)
                    if isinstance(binding, ParameterSource):
                        matching = [context for context in contexts
                                    if context.target.module == binding_module
                                    and context.target.qualname == binding.scope]
                        if (len(matching) != 1 or binding.derived
                                or binding.attributes
                                or not binding.scope.endswith('.__init__')):
                            return []
                        binding = self._bounded_argument_source(matching[0], binding.name)
                        binding_module = matching[0].caller_module
                    values = (binding.sources if isinstance(binding, SourceSet)
                              else (binding,))
                    for value in values:
                        candidates = self._local_callable_class_candidates(
                            binding_module, value, tracers.get(binding_module), tracers,
                            visited=seen)
                        if not candidates:
                            return []
                        classes.extend(candidates)
            return self._dedupe_list(classes)
        finally:
            self._callable_field_in_progress.remove(key)

    ## Reject object escapes that could overwrite an injected callable field.
    #  @param module Field-call module.
    #  @param source Parameter-backed field-call source.
    #  @param arguments Concrete receiver constructor sources.
    #  @param tracers Project analyzers.
    #  @return True when a receiver is passed elsewhere or aliased ambiguously.
    def _callable_field_receiver_escapes(self, module, source, arguments, tracers):
        names = {(module, source.parameter_scope): {source.parameter_name}}
        identities = []
        for origin, receiver in arguments:
            if not isinstance(receiver, CallResult):
                return True
            identity = self._resolve_local_class_identity(origin, receiver, tracers)
            if identity is None:
                return True
            if not self._constructor_field_method_is_readonly(
                    identity, '__init__', source.method, tracers):
                return True
            identities.append(identity)
            cg = self.project_cg.modules.get(origin)
            for edge in cg.edges if cg is not None else ():
                if (edge.call_lineno == receiver.call_lineno
                        and edge.call_col_offset == receiver.call_col_offset):
                    names.setdefault((origin, edge.caller.qualname), set()).update(
                        edge.assigned_to)

        def carries_name(node, tracked):
            if isinstance(node, ast.Name):
                return node.id in tracked
            if isinstance(node, ast.Attribute):
                return False
            return any(carries_name(child, tracked)
                       for child in ast.iter_child_nodes(node))

        for (origin, scope), tracked in names.items():
            tree = tracers[origin]._module_tree
            calls = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    calls.setdefault((node.lineno, node.col_offset), []).append(node)
            # Alias assignments may hide an escape inside a container argument.
            # Keep this constructor-only path conservative until aliases carry
            # object mutation facts of their own.
            for node in ast.walk(tree):
                if (isinstance(node, (ast.Assign, ast.AnnAssign))
                        and node.value is not None
                        and carries_name(node.value, tracked)
                        and isinstance(node.value, (ast.Name, ast.List, ast.Tuple, ast.Dict))):
                    return True
            for edge in self.project_cg.modules[origin].edges:
                if edge.caller.qualname != scope:
                    continue
                candidates = calls.get((edge.call_lineno, edge.call_col_offset), [])
                if len(candidates) > 1:
                    candidates = [node for node in candidates
                                  if ast.unparse(node.func) == edge.callee_name]
                if len(candidates) != 1:
                    return True
                call = candidates[0]
                receiver = call.func
                attributes = []
                while isinstance(receiver, ast.Attribute):
                    attributes.append(receiver.attr)
                    receiver = receiver.value
                if (isinstance(receiver, ast.Name) and receiver.id in tracked
                        and isinstance(call.func, ast.Attribute)
                        and not (isinstance(call.func.value, ast.Name)
                                 and call.func.attr == source.method)):
                    checked_method = call.func.attr
                    if len(attributes) > 1:
                        if attributes[-1] not in self._constructor_only_fields:
                            return True
                        checked_method = '__init__'
                    if not all(self._constructor_field_method_is_readonly(
                            identity, checked_method, source.method, tracers)
                            for identity in identities):
                        return True
                if not any(carries_name(arg, tracked)
                        for arg in list(call.args) + [kw.value for kw in call.keywords]):
                    continue
                if self._edge_targets_local_function(
                        edge, origin, module, source.parameter_scope,
                        tracers[origin], tracers):
                    continue
                return True
        return False

    ## Check a local method for escapes of its constructor-only field owner.
    #  @param identity Project-local class identity.
    #  @param method Method invoked on the object.
    #  @param field Constructor-injected callable field.
    #  @param tracers Project analyzers.
    #  @param seen Local method recursion guard.
    #  @return True only for inspectable non-escaping instance method bodies.
    def _constructor_field_method_is_readonly(
            self, identity, method, field, tracers, seen=None):
        key = (identity, method)
        seen = set(seen or ())
        if key in seen:
            return False
        seen.add(key)
        tree = tracers[identity[0]]._module_tree
        classes = [node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef) and node.name == identity[1]]
        if len(classes) != 1:
            return False
        methods = [node for node in classes[0].body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name in (method, '__init__')]
        method_names = {node.name for node in classes[0].body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if not any(node.name == method for node in methods):
            return False
        for function in methods:
            args = list(function.args.posonlyargs) + list(function.args.args)
            if not args or function.decorator_list:
                return False
            receiver = args[0].arg
            parents = {id(child): node for node in ast.walk(function)
                       for child in ast.iter_child_nodes(node)}
            for node in ast.walk(function):
                if (isinstance(node, ast.Name) and node.id == receiver
                        and isinstance(node.ctx, ast.Load)
                        and not isinstance(parents.get(id(node)), ast.Attribute)):
                    return False
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == receiver and node.attr in method_names):
                    parent = parents.get(id(node))
                    if not (isinstance(parent, ast.Call) and parent.func is node):
                        return False
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == receiver
                        and node.func.attr != field
                        and not (node.func.attr in self._constructor_only_fields
                                 and node.func.attr not in method_names)
                        and not self._constructor_field_method_is_readonly(
                            identity, node.func.attr, field, tracers, seen)):
                    return False
        return True

    ## Resolve local classes represented by callable-object constructor calls.
    #
    #  A call such as ``f(value)`` stores the constructor's defining module in
    #  CallEdge.callee while retaining the imported class spelling in
    #  CallEdge.callee_name.  CallResult sources carry the same spelling in
    #  display_name.  Reconstruct the class only from that existing evidence
    #  and the caller's import bindings.
    #  @param module Module containing the callable-object call.
    #  @param source Callable source or SourceSet of callable sources.
    #  @param tracer Analyzer for the caller module.
    #  @param tracers All project analyzers.
    #  @param visited Recursion guard.
    #  @param display_name Optional call-edge spelling for the source.
    #  @return List of local class identities.
    def _local_callable_class_candidates(
            self, module, source, tracer, tracers, visited=None,
            display_name=None):
        source = normalize_source(source)
        seen = set(visited or set())
        key = (type(source).__name__, source_display(source))
        if key in seen:
            return []
        seen.add(key)

        if (isinstance(source, InstanceMethod) and source.parameter_scope
                and source.receiver == source.parameter_name):
            return self._parameter_callable_field_classes(
                module, source, tracers, seen)

        if isinstance(source, SourceSet):
            candidates = []
            for item in source.sources:
                candidates.extend(self._local_callable_class_candidates(
                    module, item, tracer, tracers, set(seen)))
            return self._dedupe_list(candidates)

        direct = self._local_class_from_source(module, source)
        if direct is not None:
            return [direct]
        if isinstance(source, CallResult) and isinstance(source.callee, str):
            call_method = ".__call__"
            if source.callee.endswith(call_method):
                direct = self._local_class_from_source(
                    module, source.callee[:-len(call_method)])
                if direct is not None:
                    return [direct]
        if tracer is None:
            return []

        display = display_name
        if display is None and isinstance(source, CallResult):
            display = source.display_name
        if (not display
                and isinstance(source, CallResult)
                and isinstance(source.callee, str)):
            for imported in getattr(
                    tracer, "import_from_symbols", {}).values():
                if (isinstance(imported, str)
                        and (imported == source.callee
                             or imported.startswith(source.callee + "."))):
                    identity = self._local_class_from_source(
                        module, imported)
                    if identity is not None:
                        return [identity]
        if not isinstance(display, str) or not display:
            return []
        if "." in display:
            alias, class_name = display.rsplit(".", 1)
            bound_module = tracer.import_from_symbols.get(alias)
            if bound_module is None:
                bound_module = tracer.symbols.direct.get(alias)
        else:
            class_name = display
            bound_module = tracer.import_from_symbols.get(display)
            if bound_module is None:
                bound_module = tracer.symbols.direct.get(display)
        bound_module = normalize_source(bound_module)
        if isinstance(bound_module, CallResult):
            bound_module = bound_module.callee
        if isinstance(bound_module, str):
            imported_symbol = tracer.import_from_symbols.get(bound_module)
            if imported_symbol is not None:
                bound_module = imported_symbol
            if bound_module.endswith(".__call__"):
                bound_module = bound_module[:-len(".__call__")]
        if not isinstance(bound_module, str):
            return []
        candidate_source = (
            bound_module if "." not in display
            else bound_module + "." + class_name)
        identity = self._local_class_from_source(module, candidate_source)
        return [identity] if identity is not None else []

    ## Return whether an edge receiver may have one required local class.
    #
    #  @param edge Project call-graph edge.
    #  @param caller_module Module containing the edge.
    #  @param required Pair of required (module, class).
    #  @param tracers Dict of module name to analyzer.
    #  @return False only when concrete local class evidence excludes it.
    def _edge_receiver_may_have_class(
            self, edge, caller_module, required, tracers):
        candidates = self._local_class_candidates(
            caller_module, edge.receiver_source, tracers)
        if not candidates:
            return True
        return any(
            self._local_class_is_or_derives(
                candidate[0], candidate[1],
                required[0], required[1], tracers)
            for candidate in candidates
        )

    ## Select one positional element from a function return summary.
    #
    #  SourceSet represents alternative return branches, so every branch must
    #  provide the same positional element. A tuple DerivedResult is the only
    #  aggregate shape consumed here. Other return values remain whole-result
    #  sources and use the existing path.
    #  @param source Return summary source.
    #  @param index Zero-based assignment position.
    #  @return Selected element source, or None when position is unproven.
    def _return_item_source(self, source, index):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            items = []
            for branch in source.sources:
                item = self._return_item_source(branch, index)
                if item is None:
                    return None
                items.append(item)
            if not items:
                return None
            return make_source_set(items, origin=source.origin or "return")
        if isinstance(source, DerivedResult) and source.kind == "tuple":
            if index < 0 or index >= len(source.sources):
                return None
            return normalize_source(source.sources[index])
        return None

    ## Return whether a summary explicitly describes a positional tuple.
    #  @param source Return summary source.
    #  @return True when every alternative has tuple positional evidence.
    def _is_tuple_return_source(self, source):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            return bool(source.sources) and all(
                self._is_tuple_return_source(branch)
                for branch in source.sources)
        return isinstance(source, DerivedResult) and source.kind == "tuple"

    ## Return whether one tuple element is concrete enough to replace a
    #  caller binding. Parameter-derived and unresolved elements must stay on
    #  the legacy bounded path, because their owner still depends on a
    #  runtime argument or an external return contract.
    #  @param source Positional tuple element source.
    #  @return True when the source carries usable owner evidence.
    def _is_resolved_return_item(self, source):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            return bool(source.sources) and all(
                self._is_resolved_return_item(branch)
                for branch in source.sources)
        if isinstance(source, (UnknownSource, DerivedResult)):
            return False
        return source is not None

    ## Check whether a return summary contains an opaque call result.
    #
    #  A call's callable owner is not a contract for the returned object's
    #  owner. Inherited-target fallback therefore must not extend that legacy
    #  inference unless the CallResult carries an explicit result_source.
    #  @param source Function return summary source.
    #  @return True when any branch contains an unqualified CallResult.
    def _has_opaque_call_result(self, source):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            return any(
                self._has_opaque_call_result(branch)
                for branch in source.sources)
        if isinstance(source, CallResult):
            if source.result_source is None:
                return True
            return self._has_opaque_call_result(source.result_source)
        if isinstance(source, DerivedResult):
            return any(
                self._has_opaque_call_result(branch)
                for branch in source.sources)
        return False

    ## Check whether a return summary is an unresolved receiver method call.
    #
    #  The callable owner of ``self.worker.predict()`` does not establish the
    #  owner of the object returned by ``predict``.  Constructor-style local
    #  factories remain separate because their existing source contract is a
    #  CallResult rather than a parameter-backed InstanceMethod.
    #  @param source Function return summary source.
    #  @return True when a branch returns a parameter-backed method result.
    def _has_unresolved_method_result(self, source):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            return any(
                self._has_unresolved_method_result(branch)
                for branch in source.sources)
        if isinstance(source, InstanceMethod):
            return bool(source.parameter_scope)
        if isinstance(source, DerivedResult):
            return any(
                self._has_unresolved_method_result(branch)
                for branch in source.sources)
        return False

    ## Preserve exact result calls for assigned project-local methods.
    #
    #  Single-file analysis cannot know whether an imported class is defined
    #  by the project. Once the project graph is complete, replace only an
    #  assigned InstanceMethod source whose edge has one local target and a
    #  return summary. The call itself retains local callable ownership; only
    #  the assigned result gains call-site context.
    #
    #  @param tracers Dict of module name to analyzer.
    def _bind_bounded_local_call_results(self, tracers):
        for caller_module, module_cg in self.project_cg.modules.items():
            tracer = tracers.get(caller_module)
            if tracer is None:
                continue
            assignment_edges = {}
            for edge in module_cg.edges:
                for assigned_name in edge.assigned_to:
                    key = (edge.caller.qualname, assigned_name)
                    assignment_edges.setdefault(key, []).append(edge)
            for edges in assignment_edges.values():
                edges.sort(key=lambda item: (
                    item.call_lineno, item.call_col_offset))

            for edge in module_cg.edges:
                if not edge.assigned_to:
                    continue
                targets = self._local_edge_targets(
                    edge, caller_module, tracers)
                if len(targets) != 1:
                    continue
                target = targets[0]
                summary = self.project_cg.modules[
                    target.module].functions.get(target.qualname)
                if summary is None or summary.returns is None:
                    continue
                exact_target = self._edge_targets_local_function(
                    edge, caller_module, target.module, target.qualname,
                    tracer, tracers)
                if (not exact_target
                        and self._has_opaque_call_result(summary.returns)):
                    continue
                for assigned_index, assigned_name in enumerate(
                        edge.assigned_to):
                    key = (edge.caller.qualname, assigned_name)
                    assigned_edges = assignment_edges.get(key, [])
                    edge_index = assigned_edges.index(edge)
                    next_position = None
                    if edge_index + 1 < len(assigned_edges):
                        next_edge = assigned_edges[edge_index + 1]
                        next_position = (
                            next_edge.call_lineno,
                            next_edge.call_col_offset,
                        )
                    current = normalize_source(
                        tracer.symbols.direct.get(assigned_name))
                    result_source = None
                    if len(edge.assigned_to) > 1:
                        selected = self._return_item_source(
                            summary.returns, assigned_index)
                        if (selected is not None
                                and self._is_resolved_return_item(selected)):
                            result_source = selected
                        elif self._is_tuple_return_source(summary.returns):
                            # A positional tuple contract exists but this
                            # element cannot be proven. Keep the pre-existing
                            # bounded path instead of replacing it with an
                            # unresolved aggregate result.
                            continue
                    result_call = CallResult(
                        target.module + "." + target.qualname,
                        display_name=edge.callee_name,
                        call_lineno=edge.call_lineno,
                        call_col_offset=edge.call_col_offset,
                        result_source=result_source,
                    )
                    current_position = (
                        getattr(current, "call_lineno", 0),
                        getattr(current, "call_col_offset", 0),
                    )
                    is_latest_edge = edge_index == len(assigned_edges) - 1
                    if (is_latest_edge
                            and (isinstance(current, InstanceMethod)
                                 or current_position == (
                                     edge.call_lineno,
                                     edge.call_col_offset))):
                        tracer.symbols.direct[assigned_name] = result_call
                    for call_record in tracer.api_calls:
                        func_name = call_record.get("func_name", "")
                        record_base = normalize_source(
                            call_record.get("base"))
                        receiver_matches = record_base == current
                        if (isinstance(record_base, InstanceMethod)
                                and normalize_source(record_base.receiver)
                                == assigned_name):
                            receiver_matches = True
                        if (isinstance(record_base, InstanceMethod)
                                and normalize_source(record_base.receiver)
                                == current):
                            receiver_matches = True
                        if (isinstance(record_base, InstanceMethod)
                                and isinstance(record_base.receiver, str)
                                and isinstance(result_call.callee, str)):
                            receiver_matches = (
                                record_base.receiver
                                == result_call.callee.rsplit(".", 1)[-1])
                        if (isinstance(current, InstanceMethod)
                                and isinstance(record_base, InstanceMethod)
                                and current.method == record_base.method
                                and current.method
                                == target.qualname.rsplit(".", 1)[-1]):
                            receiver_matches = True
                        caller_scope = (
                            "" if edge.caller.qualname == "<module>"
                            else edge.caller.qualname)
                        record_position = (
                            call_record.get("lineno", 0),
                            call_record.get("col_offset", 0),
                        )
                        if (receiver_matches
                                and func_name.startswith(assigned_name + ".")
                                and call_record.get("scope_name", "")
                                == caller_scope
                                and record_position > (
                                    edge.call_lineno,
                                    edge.call_col_offset)
                                and (next_position is None
                                     or record_position <= next_position)):
                            call_record["base"] = InstanceMethod(
                                result_call,
                                func_name.rsplit(".", 1)[-1],
                            )
                    for future_edge in module_cg.edges:
                        if future_edge.caller != edge.caller:
                            continue
                        future_position = (
                            future_edge.call_lineno,
                            future_edge.call_col_offset)
                        if (future_position <= (
                                edge.call_lineno, edge.call_col_offset)
                                or (next_position is not None
                                    and future_position > next_position)):
                            continue
                        if (not isinstance(future_edge.callee_name, str)
                                or not future_edge.callee_name.startswith(
                                    assigned_name + ".")):
                            continue
                        future_edge.receiver_source = result_call
                        if isinstance(future_edge.callee, InstanceMethod):
                            future_edge.callee = InstanceMethod(
                                result_call, future_edge.callee.method)
                        future_edge.callee_source = result_call

    ## Bind one converged local callback return to a Pool.map result field.
    #
    #  The existing callback edge proves which project function is invoked.
    #  This pass adds the reverse result edge: every callback return branch
    #  must resolve to one owner before calls through a mapped result item are
    #  rewritten. The rewrite ends at the next source-level rebind.
    #  @param tracers Dict of module name to analyzer.
    def _bind_bounded_callback_map_results(self, tracers):
        for caller_module, module_cg in self.project_cg.modules.items():
            tracer = tracers.get(caller_module)
            if tracer is None:
                continue
            for edge in module_cg.edges:
                if (not edge.assigned_to
                        or not self._is_multiprocessing_map_edge(
                            edge, tracer)):
                    continue
                callback_names = self._dedupe_list(
                    getattr(edge, "callback_args", {}).values())
                if len(callback_names) != 1:
                    continue
                callback_summary = module_cg.functions.get(
                    callback_names[0])
                if (callback_summary is None
                        or callback_summary.returns is None):
                    continue
                owners = self._dedupe_list(
                    self._origin_candidates(
                        caller_module, callback_summary.returns, tracers))
                owners = [
                    owner for owner in owners
                    if owner not in (None, "")
                ]
                result_source = callback_summary.returns
                if len(owners) != 1 or owners[0] == "unknown":
                    result_source = UnknownSource(
                        "mixed callback result owners")
                for assigned_name in edge.assigned_to:
                    next_position = self._next_assignment_rebind_position(
                        tracer, module_cg, edge, assigned_name)
                    self._rewrite_mapped_result_records(
                        tracer, edge, assigned_name,
                        result_source, next_position)
                    self._rewrite_mapped_result_edges(
                        module_cg, edge, assigned_name,
                        result_source, next_position)
                    class_name = self._stable_field_assignment_class(
                        tracer, module_cg, edge, assigned_name)
                    if class_name:
                        self._rewrite_class_mapped_result_records(
                            tracer, module_cg, class_name, assigned_name,
                            result_source)
                        self._rewrite_class_mapped_result_edges(
                            module_cg, class_name, assigned_name,
                            result_source)

    ## Return the owning class for one uniquely assigned instance field.
    #  @param tracer Analyzer containing field assignment references.
    #  @param module_cg Caller module call graph.
    #  @param edge Pool.map assignment edge.
    #  @param assigned_name Candidate self-field name.
    #  @return Class name when the field has exactly one assignment, else "".
    def _stable_field_assignment_class(
            self, tracer, module_cg, edge, assigned_name):
        if (not assigned_name.startswith("self.")
                or "." not in edge.caller.qualname):
            return ""
        class_name = edge.caller.qualname.split(".", 1)[0]
        if class_name not in module_cg.classes:
            return ""
        for ref in getattr(tracer, "symbol_refs", []):
            if ref.symbol != assigned_name or ref.kind != "variable":
                continue
            source = normalize_source(ref.source)
            if (not isinstance(source, CallResult)
                    or source.call_lineno != edge.call_lineno
                    or source.call_col_offset != edge.call_col_offset):
                return ""
        for other_edge in module_cg.edges:
            if (other_edge is not edge
                    and assigned_name in other_edge.assigned_to):
                return ""
        return class_name

    ## Find the next source-level assignment to one mapped result binding.
    #  @param tracer Analyzer containing provenance references.
    #  @param module_cg Caller module call graph.
    #  @param edge Current Pool.map call edge.
    #  @param assigned_name Name or self-field receiving the result.
    #  @return Position tuple or None.
    def _next_assignment_rebind_position(
            self, tracer, module_cg, edge, assigned_name):
        current = (edge.call_lineno, edge.call_col_offset)
        scopes = {
            edge.caller.qualname,
            edge.caller.qualname.rsplit(".", 1)[-1],
        }
        candidates = []
        for ref in getattr(tracer, "symbol_refs", []):
            position = (
                getattr(ref, "lineno", 0),
                getattr(ref, "col_offset", 0),
            )
            if (ref.symbol == assigned_name
                    and getattr(ref, "scope_name", "") in scopes
                    and position > current):
                candidates.append(position)
        for future_edge in module_cg.edges:
            position = (
                future_edge.call_lineno,
                future_edge.call_col_offset,
            )
            if (future_edge.caller == edge.caller
                    and assigned_name in future_edge.assigned_to
                    and position > current):
                candidates.append(position)
        return min(candidates) if candidates else None

    ## Return the method suffix following a mapped container item.
    #  @param func_name Callable spelling from an API record or edge.
    #  @param assigned_name Mapped result binding.
    #  @return Empty string for direct item calls, method name for item methods,
    #          or None when the callable is unrelated.
    @staticmethod
    def _mapped_item_method(func_name, assigned_name):
        if (not isinstance(func_name, str)
                or not func_name.startswith(assigned_name + "[")):
            return None
        closing = func_name.find("]", len(assigned_name) + 1)
        if closing < 0:
            return None
        suffix = func_name[closing + 1:]
        if not suffix:
            return ""
        if not suffix.startswith("."):
            return None
        return suffix.rsplit(".", 1)[-1]

    ## Rewrite API records using one bounded mapped-result item source.
    #  @param tracer Caller analyzer.
    #  @param edge Pool.map assignment edge.
    #  @param assigned_name Mapped result binding.
    #  @param result_source Converged callback return summary.
    #  @param next_position Next rebind position or None.
    def _rewrite_mapped_result_records(
            self, tracer, edge, assigned_name, result_source,
            next_position):
        caller_scope = ("" if edge.caller.qualname == "<module>"
                        else edge.caller.qualname.rsplit(".", 1)[-1])
        start = (edge.call_lineno, edge.call_col_offset)
        for record in tracer.api_calls:
            position = (
                record.get("lineno", 0),
                record.get("col_offset", 0),
            )
            method = self._mapped_item_method(
                record.get("func_name", ""), assigned_name)
            if (method is None
                    or record.get("scope_name", "") != caller_scope
                    or position <= start
                    or (next_position is not None
                        and position >= next_position)):
                continue
            record["base"] = (
                result_source if not method
                else InstanceMethod(result_source, method))

    ## Rewrite downstream call edges using a mapped-result item source.
    #  @param module_cg Caller module call graph.
    #  @param edge Pool.map assignment edge.
    #  @param assigned_name Mapped result binding.
    #  @param result_source Converged callback return summary.
    #  @param next_position Next rebind position or None.
    def _rewrite_mapped_result_edges(
            self, module_cg, edge, assigned_name, result_source,
            next_position):
        start = (edge.call_lineno, edge.call_col_offset)
        for future_edge in module_cg.edges:
            position = (
                future_edge.call_lineno,
                future_edge.call_col_offset,
            )
            method = self._mapped_item_method(
                future_edge.callee_name, assigned_name)
            if (method is None
                    or future_edge.caller != edge.caller
                    or position <= start
                    or (next_position is not None
                        and position >= next_position)):
                continue
            source = (
                result_source if not method
                else InstanceMethod(result_source, method))
            future_edge.callee = source
            future_edge.callee_source = source
            if method:
                future_edge.receiver_source = result_source

    ## Resolve the unique class containing one API record's method scope.
    #  @param module_cg Module call graph.
    #  @param scope_name Bare method scope stored on the API record.
    #  @return Class name when unique, otherwise "".
    @staticmethod
    def _record_scope_class(module_cg, scope_name):
        matches = [
            class_name for class_name, summary in module_cg.classes.items()
            if scope_name in summary.methods
        ]
        return matches[0] if len(matches) == 1 else ""

    ## Rewrite mapped field-item records across one proven local class.
    #  @param tracer Caller analyzer.
    #  @param module_cg Caller module call graph.
    #  @param class_name Class with one field assignment.
    #  @param assigned_name Mapped self-field binding.
    #  @param result_source Converged callback return summary.
    def _rewrite_class_mapped_result_records(
            self, tracer, module_cg, class_name, assigned_name,
            result_source):
        for record in tracer.api_calls:
            method = self._mapped_item_method(
                record.get("func_name", ""), assigned_name)
            if (method is None
                    or self._record_scope_class(
                        module_cg, record.get("scope_name", ""))
                    != class_name):
                continue
            record["base"] = (
                result_source if not method
                else InstanceMethod(result_source, method))

    ## Rewrite mapped field-item edges across one proven local class.
    #  @param module_cg Caller module call graph.
    #  @param class_name Class with one field assignment.
    #  @param assigned_name Mapped self-field binding.
    #  @param result_source Converged callback return summary.
    def _rewrite_class_mapped_result_edges(
            self, module_cg, class_name, assigned_name, result_source):
        prefix = class_name + "."
        for edge in module_cg.edges:
            method = self._mapped_item_method(
                edge.callee_name, assigned_name)
            if method is None or not edge.caller.qualname.startswith(prefix):
                continue
            source = (
                result_source if not method
                else InstanceMethod(result_source, method))
            edge.callee = source
            edge.callee_source = source
            if method:
                edge.receiver_source = result_source

    ## Bind a local generator or returned iterable to its exact for-loop.
    #  The binding is bounded by the exact iterator call site and the next
    #  rebind of each loop target. No external return contract is inferred.
    #  @param tracers Dict of module name to analyzer.
    def _bind_bounded_local_iteration_results(self, tracers):
        for caller_module, module_cg in self.project_cg.modules.items():
            tracer = tracers.get(caller_module)
            if tracer is None:
                continue
            for binding in getattr(module_cg, "iteration_bindings", []):
                if binding.caller.module != caller_module:
                    continue
                iterator_edges = [
                    edge for edge in module_cg.edges
                    if edge.caller == binding.caller
                    and edge.call_lineno == binding.call_lineno
                    and edge.call_col_offset == binding.call_col_offset
                    and edge.callee_name == binding.callee_name
                ]
                if len(iterator_edges) != 1:
                    continue
                edge = iterator_edges[0]
                targets = self._local_edge_targets(
                    edge, caller_module, tracers)
                if len(targets) != 1:
                    continue
                summary = self.project_cg.modules[
                    targets[0].module].functions.get(targets[0].qualname)
                if summary is None:
                    continue
                if summary.yields is not None:
                    yield_source = self._substitute_generator_parameters(
                        summary.yields, edge, summary, tracers)
                else:
                    # Preserve the selected call until its returned elements
                    # can be substituted, separately from the container owner.
                    yield_source = ContainerIter(CallResult(
                        edge.callee_source,
                        display_name=edge.callee_name,
                        call_lineno=edge.call_lineno,
                        call_col_offset=edge.call_col_offset,
                        source_module=caller_module))
                if yield_source is None:
                    continue
                for target_name in binding.target_names:
                    next_position = self._next_iteration_rebind_position(
                        module_cg, tracer, binding, target_name)
                    self._rewrite_generator_target_records(
                        tracer, binding, target_name, yield_source,
                        next_position)
                    self._rewrite_generator_target_edges(
                        module_cg, binding, target_name, yield_source,
                        next_position)
                    current = normalize_source(
                        tracer.symbols.direct.get(target_name))
                    if (isinstance(current, CallResult)
                            and current.call_lineno == binding.call_lineno
                            and current.call_col_offset
                            == binding.call_col_offset):
                        tracer.symbols.direct[target_name] = yield_source

    ## Substitute exact call-edge arguments into a local generator yield.
    #
    #  A generator summary may yield one of its own parameters.  The summary
    #  is declaration-level evidence, so it still names the generator's
    #  ParameterSource.  An exact local call edge supplies the concrete source
    #  for that parameter.  This substitution is bounded to one iterator call
    #  site and does not infer any external library return semantics.
    #  @param source Generator yield source or nested source IR.
    #  @param edge Exact call edge for the iterator expression.
    #  @param summary Callee function summary containing parameter metadata.
    #  @return Substituted source, or None when the parameter is ambiguous.
    def _substitute_generator_parameters(
            self, source, edge, summary, tracers, _seen=None):
        source = normalize_source(source)
        if isinstance(source, ParameterSource):
            if source.scope != summary.id.qualname:
                nested = self._nested_generator_parameter_argument(
                    source, edge, summary, tracers, _seen)
                return source if nested is None else nested
            parameter_index = summary.params.index(source.name) \
                if source.name in summary.params else None
            if parameter_index is None:
                return None
            arguments = self._edge_parameter_sources(
                edge, summary, source.name, parameter_index,
                prefer_protocol_shape=True)
            if arguments is None or len(arguments) != 1:
                return None
            argument = normalize_source(arguments[0])
            if source.attributes or source.derived:
                # The current bounded generator contract preserves direct
                # parameter yields. Attribute and derived yields need a
                # separate source-composition contract.
                return None
            return argument
        if isinstance(source, SourceSet):
            substituted = []
            for item in source.sources:
                item_source = self._substitute_generator_parameters(
                    item, edge, summary, tracers, _seen)
                if item_source is None:
                    return None
                substituted.append(item_source)
            return make_source_set(substituted, origin=source.origin)
        if isinstance(source, InstanceMethod):
            receiver = self._substitute_generator_parameters(
                source.receiver, edge, summary, tracers, _seen)
            if receiver is None:
                return None
            return InstanceMethod(
                receiver, source.method, source.parameter_scope,
                source.parameter_name)
        if isinstance(source, ContainerItem):
            container = self._substitute_generator_parameters(
                source.container, edge, summary, tracers, _seen)
            if container is None:
                return None
            return ContainerItem(container, source.index)
        if isinstance(source, ContainerIter):
            container = self._substitute_generator_parameters(
                source.container, edge, summary, tracers, _seen)
            if container is None:
                return None
            return ContainerIter(container)
        if isinstance(source, CallResult):
            callee = self._substitute_generator_parameters(
                source.callee, edge, summary, tracers, _seen)
            if callee is None:
                return None
            result_source = source.result_source
            if result_source is not None:
                result_source = self._substitute_generator_parameters(
                    result_source, edge, summary, tracers, _seen)
                if result_source is None:
                    return None
            return CallResult(
                callee,
                source.display_name,
                source.call_lineno,
                source.call_col_offset,
                source.source_module,
                result_source)
        if isinstance(source, DerivedResult):
            operands = []
            for item in source.sources:
                item_source = self._substitute_generator_parameters(
                    item, edge, summary, tracers, _seen)
                if item_source is None:
                    return None
                operands.append(item_source)
            return DerivedResult(source.kind, tuple(operands), source.attribute)
        return source

    ## Resolve a parameter inherited through one local yield-from edge.
    #
    #  For ``outer(value): yield from inner(value)``, the return summary of
    #  outer may still contain ``ParameterSource('inner', 'value')``.  Follow
    #  that exact local edge once, then let the caller's edge resolve the
    #  resulting outer parameter.  Ambiguous targets remain unresolved.
    #  @param source Nested generator ParameterSource.
    #  @param edge Outer iterator call edge.
    #  @param summary Current generator summary.
    #  @param tracers Per-module analyzers.
    #  @param _seen Recursion guard.
    #  @return Substituted source, or None when no unique local edge exists.
    def _nested_generator_parameter_argument(
            self, source, edge, summary, tracers, _seen=None):
        seen = set(_seen or set())
        key = (summary.id.module, summary.id.qualname,
               source.scope, source.name)
        if key in seen:
            return None
        seen.add(key)
        module = summary.id.module
        module_cg = self.project_cg.modules.get(module)
        if module_cg is None:
            return None
        nested_edges = []
        nested_summary = None
        for nested_edge in module_cg.edges:
            if nested_edge.caller.qualname != summary.id.qualname:
                continue
            targets = self._local_edge_targets(
                nested_edge, module, tracers)
            matching = [
                target for target in targets
                if target.qualname == source.scope
            ]
            if len(matching) != 1:
                continue
            candidate = module_cg.functions.get(source.scope)
            if candidate is None:
                candidate = self.project_cg.modules[
                    matching[0].module].functions.get(matching[0].qualname)
            if candidate is None or source.name not in candidate.params:
                continue
            nested_edges.append((nested_edge, candidate))
            nested_summary = candidate
        if len(nested_edges) != 1 or nested_summary is None:
            return None
        parameter_index = nested_summary.params.index(source.name)
        arguments = self._edge_parameter_sources(
            nested_edges[0][0], nested_summary, source.name,
            parameter_index, prefer_protocol_shape=True)
        if arguments is None or len(arguments) != 1:
            return None
        return self._substitute_generator_parameters(
            arguments[0], edge, summary, tracers, seen)

    ## Find the next source-level rebind of one loop target.
    #  @param module_cg Module call graph.
    #  @param binding Current iteration binding.
    #  @param target_name Loop target name.
    #  @return Position tuple or None.
    def _next_iteration_rebind_position(self, module_cg, tracer, binding,
                                        target_name):
        current = (binding.call_lineno, binding.call_col_offset)
        candidates = []
        for edge in module_cg.edges:
            if edge.caller != binding.caller or target_name not in (
                    edge.assigned_to or []):
                continue
            position = (edge.call_lineno, edge.call_col_offset)
            if position > current:
                candidates.append(position)
        for other in getattr(module_cg, "iteration_bindings", []):
            if other is binding or other.caller != binding.caller:
                continue
            if target_name not in other.target_names:
                continue
            position = (other.call_lineno, other.call_col_offset)
            if position > current:
                candidates.append(position)
        caller_scope = binding.caller.qualname
        caller_short_scope = caller_scope.rsplit(".", 1)[-1]
        for ref in getattr(tracer, "symbol_refs", []):
            if ref.symbol != target_name:
                continue
            ref_scope = getattr(ref, "scope_name", "")
            if ref_scope not in ("", caller_scope, caller_short_scope):
                continue
            position = (getattr(ref, "lineno", 0),
                        getattr(ref, "col_offset", 0))
            if position > current:
                candidates.append(position)
        return min(candidates) if candidates else None

    ## Rewrite API call records whose receiver is one generator target.
    #  @param tracer Analyzer for the caller module.
    #  @param binding Iteration binding.
    #  @param target_name Loop target name.
    #  @param yield_source Source yielded by the target function.
    #  @param next_position Next rebind position, if any.
    def _rewrite_generator_target_records(
            self, tracer, binding, target_name, yield_source,
            next_position):
        caller_scope = ("" if binding.caller.qualname == "<module>"
                        else binding.caller.qualname)
        start = (binding.call_lineno, binding.call_col_offset)
        for record in tracer.api_calls:
            position = (record.get("lineno", 0),
                        record.get("col_offset", 0))
            func_name = record.get("func_name", "")
            if (record.get("scope_name", "") != caller_scope
                    or position <= start
                    or (next_position is not None
                        and position > next_position)
                    or not func_name.startswith(target_name + ".")):
                continue
            record["base"] = InstanceMethod(
                yield_source, func_name.rsplit(".", 1)[-1])

    ## Rewrite future call edges whose receiver is one generator target.
    #  @param module_cg Module call graph.
    #  @param binding Iteration binding.
    #  @param target_name Loop target name.
    #  @param yield_source Source yielded by the target function.
    #  @param next_position Next rebind position, if any.
    def _rewrite_generator_target_edges(
            self, module_cg, binding, target_name, yield_source,
            next_position):
        start = (binding.call_lineno, binding.call_col_offset)
        for edge in module_cg.edges:
            position = (edge.call_lineno, edge.call_col_offset)
            if (edge.caller != binding.caller
                    or position <= start
                    or (next_position is not None
                        and position > next_position)
                    or not edge.callee_name.startswith(target_name + ".")):
                continue
            edge.receiver_source = yield_source
            edge.callee = InstanceMethod(
                yield_source, edge.callee_name.rsplit(".", 1)[-1])
            edge.callee_source = edge.callee

    ## Propagate an explicitly proven receiver through an assigned method call.
    #
    #  This is the next bounded call-graph step after a local function return:
    #  ``left_view = left.reshape(...)`` may carry the owner already proven for
    #  ``left`` into ``left_view``. Only an existing result-owner contract is
    #  accepted; an unresolved receiver or method remains unchanged.
    #  @param tracers Dict of module name to analyzer.
    def _bind_proven_result_method_results(self, tracers):
        for module, module_cg in self.project_cg.modules.items():
            tracer = tracers.get(module)
            if tracer is None:
                continue
            edges = sorted(
                module_cg.edges,
                key=lambda edge: (
                    edge.caller.qualname,
                    edge.call_lineno,
                    edge.call_col_offset),
            )
            flow = {}
            for edge in edges:
                if not isinstance(edge.callee_name, str):
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                root, separator, _ = edge.callee_name.partition(".")
                if not separator:
                    # A non-method assignment is a rebinding barrier.  The
                    # final module symbol table is intentionally not used as
                    # a time-insensitive fallback here.
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                flow_key = (edge.caller.qualname, root)
                receiver = normalize_source(flow.get(flow_key))
                if receiver is None:
                    receiver = normalize_source(edge.receiver_source)
                if not isinstance(receiver, CallResult):
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                if receiver.result_source is None:
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                origin_module = module
                if (isinstance(receiver.callee, str)
                        and "." in receiver.callee):
                    candidate_module = receiver.callee.rsplit(".", 1)[0]
                    if candidate_module in tracers:
                        origin_module = candidate_module
                receiver_owners = self._origin_candidates(
                    origin_module, receiver.result_source, tracers)
                if (len(receiver_owners) != 1
                        or receiver_owners[0] in (
                            "local", "python", "unknown", "")):
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                return_proven = _has_return_provenance(
                    receiver.result_source)
                # Existing external result contracts are sufficient for the
                # legacy resolver, but they are not a new interprocedural
                # fact. Only local return/tuple evidence may rewrite a
                # downstream call record in this pass.
                if not return_proven:
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                method_records = [
                    record for record in tracer.api_calls
                    if record.get("lineno") == edge.call_lineno
                    and record.get("col_offset") == edge.call_col_offset
                    and record.get("scope_name", "") == (
                        "" if edge.caller.qualname == "<module>"
                        else edge.caller.qualname)
                    and record.get("func_name", "").startswith(root + ".")
                ]
                if not method_records:
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                method_name = max(
                    (record.get("func_name", "")
                     for record in method_records),
                    key=len).rsplit(".", 1)[-1]
                method_source = InstanceMethod(receiver, method_name)
                edge.receiver_source = receiver
                edge.callee = method_source
                edge.callee_source = method_source
                for record in method_records:
                    method = record.get("func_name", "").rsplit(".", 1)[-1]
                    record["base"] = InstanceMethod(receiver, method)
                if not edge.assigned_to or not _has_result_owner_contract(
                        method_name):
                    for assigned_name in edge.assigned_to:
                        flow.pop((edge.caller.qualname, assigned_name), None)
                    continue
                result_call = CallResult(
                    method_source,
                    display_name=edge.callee_name,
                    call_lineno=edge.call_lineno,
                    call_col_offset=edge.call_col_offset,
                    result_source=DerivedResult(
                        "method_result", (method_source,), method_name),
                )
                for assigned_name in edge.assigned_to:
                    flow[(edge.caller.qualname, assigned_name)] = result_call
                    tracer.symbols.direct[assigned_name] = result_call

    ## Find all project-local functions reached by one call edge.
    #
    #  @param edge Call edge to resolve.
    #  @param caller_module Module containing the edge.
    #  @param tracers Dict of module name to analyzer.
    #  @return List of matching FunctionId values.
    def _local_edge_targets(self, edge, caller_module, tracers):
        if getattr(edge, "mapping_targets", None) is not None:
            if not edge.mapping_targets_complete:
                return []
            return list(edge.mapping_targets)
        targets = []
        caller_tracer = tracers.get(caller_module)
        for target_module, module_cg in self.project_cg.modules.items():
            for qualname, summary in module_cg.functions.items():
                if self._edge_targets_local_function(
                        edge, caller_module, target_module, qualname,
                        caller_tracer, tracers):
                    targets.append(summary.id)
        # An exact subclass method wins over inherited implementations. When
        # no exact method exists, resolve the nearest available local base
        # implementation through the class graph. Multiple inherited targets
        # remain explicit and therefore cannot drive bounded result binding.
        if not targets:
            for target_module, module_cg in self.project_cg.modules.items():
                for qualname, summary in module_cg.functions.items():
                    if self._edge_targets_local_function(
                            edge, caller_module, target_module, qualname,
                            caller_tracer, tracers,
                            allow_inherited_dispatch=True):
                        targets.append(summary.id)
            targets = self._nearest_inherited_method_targets(
                targets, tracers)
        unique = []
        seen = set()
        for target in targets:
            key = (target.module, target.qualname)
            if key not in seen:
                seen.add(key)
                unique.append(target)
        return unique

    ## Remove ancestor methods hidden by a more-derived local implementation.
    #
    #  Multiple-inheritance siblings remain separate candidates because this
    #  bounded resolver does not claim a complete Python MRO. A linear local
    #  inheritance chain, however, has one unambiguous nearest implementation.
    #  @param targets Candidate inherited FunctionId values.
    #  @param tracers Dict of module name to analyzer.
    #  @return Candidate list with shadowed ancestor methods removed.
    def _nearest_inherited_method_targets(self, targets, tracers):
        class_targets = {}
        for target in targets:
            module_cg = self.project_cg.modules.get(target.module)
            parts = target.qualname.rsplit(".", 1)
            if (module_cg is not None and len(parts) == 2
                    and parts[0] in module_cg.classes
                    and parts[1] in module_cg.classes[parts[0]].methods):
                class_targets[target] = (target.module, parts[0])

        nearest = []
        for target in targets:
            target_class = class_targets.get(target)
            if target_class is None:
                nearest.append(target)
                continue
            shadowed = any(
                other != target
                and self._local_class_is_or_derives(
                    other_class[0], other_class[1],
                    target_class[0], target_class[1], tracers)
                for other, other_class in class_targets.items()
            )
            if not shadowed:
                nearest.append(target)
        return nearest

    ## Resolve every branch of an explicit local callable SourceSet.
    #
    #  This deliberately rejects inferred method-name candidates. Multiple
    #  targets are safe to converge only when the call edge itself retains
    #  each exact project-local callable selected by source control flow.
    #  @param edge Call edge containing callable provenance.
    #  @param caller_module Module containing the edge.
    #  @return List of exact FunctionId values, or an empty list.
    def _explicit_local_callable_targets(self, edge, caller_module):
        if getattr(edge, "mapping_targets", None) is not None:
            return (list(edge.mapping_targets)
                    if edge.mapping_targets_complete else [])
        source = normalize_source(getattr(edge, "callee_source", None))
        if not isinstance(source, SourceSet):
            return []
        targets = []
        seen = set()
        for branch in source.sources:
            branch = normalize_source(branch)
            if not isinstance(branch, str):
                return []
            matches = []
            for module, module_cg in self.project_cg.modules.items():
                for qualname, summary in module_cg.functions.items():
                    qualified = module + "." + qualname
                    if branch == qualified or (
                            module == caller_module
                            and branch == qualname):
                        matches.append(summary.id)
            if len(matches) != 1:
                return []
            target = matches[0]
            key = (target.module, target.qualname)
            if key not in seen:
                seen.add(key)
                targets.append(target)
        return targets

    ## Build all project-local contexts from one exact call-site position.
    #
    #  Multiple contexts are retained for a statically enumerated callable
    #  branch. Their result owners are converged by the caller.
    #  @param caller_module Module containing the call.
    #  @param call_lineno Call line number.
    #  @param call_col_offset Call column offset.
    #  @param tracers Dict of module name to analyzer.
    #  @param parent Enclosing forwarding context.
    #  @param callee_name Optional syntactic callee to distinguish nested calls
    #  sharing their start position.
    #  @return List of CallContext values.
    def _bounded_call_contexts(self, caller_module, call_lineno,
                               call_col_offset, tracers, parent=None,
                               callee_name=None):
        module_cg = self.project_cg.modules.get(caller_module)
        if module_cg is None or not call_lineno:
            return []
        edges = [
            edge for edge in module_cg.edges
            if edge.call_lineno == call_lineno
            and edge.call_col_offset == call_col_offset
        ]
        if len(edges) > 1 and callee_name:
            edges = [edge for edge in edges
                     if edge.callee_name == callee_name]
        if len(edges) != 1:
            return []
        targets = self._local_edge_targets(edges[0], caller_module, tracers)
        if len(targets) > 1:
            targets = self._explicit_local_callable_targets(
                edges[0], caller_module)
            if len(targets) <= 1:
                return []
        return [
            CallContext(
                caller_module=caller_module,
                target=target,
                edge=edges[0],
                parent=parent,
            )
            for target in targets
        ]

    ## Build one exact bounded context from a call-site position.
    #
    #  @param caller_module Module containing the call.
    #  @param call_lineno Call line number.
    #  @param call_col_offset Call column offset.
    #  @param tracers Dict of module name to analyzer.
    #  @param parent Enclosing forwarding context.
    #  @return CallContext when edge and target are unique, otherwise None.
    def _bounded_call_context(self, caller_module, call_lineno,
                              call_col_offset, tracers, parent=None):
        contexts = self._bounded_call_contexts(
            caller_module, call_lineno, call_col_offset, tracers,
            parent=parent)
        if len(contexts) != 1:
            return None
        return contexts[0]

    ## Read the argument supplied to one parameter in a bounded context.
    #
    #  @param context Exact local call context.
    #  @param parameter Target parameter name.
    #  @return Argument source or None.
    def _bounded_argument_source(self, context, parameter):
        module_cg = self.project_cg.modules.get(context.target.module)
        summary = (
            module_cg.functions.get(context.target.qualname)
            if module_cg is not None else None)
        if summary is None or parameter not in summary.params:
            return None
        keyword_args = context.edge.arg_sources.get("kw", {})
        if parameter in keyword_args:
            return keyword_args[parameter]
        if parameter == summary.vararg or parameter == summary.kwarg:
            # A variadic parameter is a pack, not one source.  It is resolved
            # only when a later edge selects one item from the pack.
            return None
        positional_params = list(getattr(summary, "positional_params", []))
        if not positional_params:
            positional_params = [
                name for name in summary.params
                if name not in (summary.vararg, summary.kwarg)
            ]
        if parameter in positional_params:
            index = positional_params.index(parameter)
            positional = context.edge.arg_sources.get("pos", {})
            if index in positional:
                return positional[index]
            star_source = self._star_positional_item_source(
                context.edge, summary, index)
            if star_source is not None:
                return star_source
        star_kwargs = getattr(context.edge, "star_kwarg_sources", [])
        if star_kwargs:
            if len(star_kwargs) != 1:
                return None
            return ContainerItem(star_kwargs[0], parameter)
        defaults = getattr(summary, "defaults", {})
        if parameter in defaults:
            return defaults[parameter]
        return None

    ## Resolve one positional parameter from a starred call argument.
    #  @param edge Call graph edge.
    #  @param summary Callee signature summary.
    #  @param index Zero-based positional parameter index.
    #  @return ContainerItem selecting the pack item, or None.
    def _star_positional_item_source(self, edge, summary, index):
        raw_stars = getattr(edge, "star_arg_sources", {})
        if any(start is None for start in raw_stars):
            return None
        stars = sorted(raw_stars.items(), key=lambda item: item[0])
        if not stars:
            return None
        matches = []
        for start, source in stars:
            if start <= index:
                matches.append((start, source))
        if len(matches) != 1:
            return None
        start, source = matches[0]
        return ContainerItem(source, index - start)

    ## Resolve one selected item from a local variadic parameter under a
    #  bounded call context.
    #  @param context Current exact call context.
    #  @param pack_source ParameterSource naming *args or **kwargs.
    #  @param index Integer position or keyword name.
    #  @return (caller module, source, parent context), or None.
    def _bounded_pack_item_source(self, context, pack_source, index):
        if not isinstance(pack_source, ParameterSource):
            return None
        current = context
        while current is not None:
            if pack_source.scope != current.target.qualname:
                current = current.parent
                continue
            module_cg = self.project_cg.modules.get(current.target.module)
            summary = (
                module_cg.functions.get(current.target.qualname)
                if module_cg is not None else None)
            if summary is None:
                return None
            edge = current.edge
            if pack_source.name == summary.vararg:
                if not isinstance(index, int):
                    return None
                positional_params = list(
                    getattr(summary, "positional_params", []))
                actual_index = len(positional_params) + index
                positional = edge.arg_sources.get("pos", {})
                if actual_index in positional:
                    return (current.caller_module,
                            positional[actual_index], current.parent)
                star_item = self._star_positional_item_source(
                    edge, summary, actual_index)
                if star_item is not None:
                    return (current.caller_module,
                            star_item, current.parent)
                return None
            if pack_source.name == summary.kwarg:
                if not isinstance(index, str):
                    return None
                keyword_args = edge.arg_sources.get("kw", {})
                if index in keyword_args:
                    return (current.caller_module,
                            keyword_args[index], current.parent)
                star_kwargs = getattr(edge, "star_kwarg_sources", [])
                if len(star_kwargs) == 1:
                    return (current.caller_module,
                            ContainerItem(star_kwargs[0], index),
                            current.parent)
                return None
            return None
        return None

    ## Resolve a constructor parameter referenced by a method return summary.
    #
    #  For Holder.expose() returning the value stored by Holder.__init__,
    #  use the exact receiver constructor edge. No class-name or library-name
    #  table is involved.
    #
    #  @param context Current method-call context.
    #  @param source Constructor ParameterSource.
    #  @param tracers Dict of module name to analyzer.
    #  @return Candidate owner strings or None when not applicable.
    def _constructor_parameter_candidates(self, context, source, tracers,
                                          _seen):
        if not isinstance(source, ParameterSource):
            return None
        if not source.scope.endswith(".__init__"):
            return None
        class_name = source.scope.rsplit(".", 1)[0]
        method_class = (
            context.target.qualname.rsplit(".", 1)[0]
            if "." in context.target.qualname else "")
        if class_name != method_class:
            return None
        receiver_source = normalize_source(context.edge.receiver_source)
        if not isinstance(receiver_source, CallResult):
            return ["unknown"]
        ctor_context = self._bounded_call_context(
            context.caller_module,
            receiver_source.call_lineno,
            receiver_source.call_col_offset,
            tracers,
            parent=context.parent,
        )
        if (ctor_context is None
                or ctor_context.target.module != context.target.module
                or ctor_context.target.qualname != source.scope):
            return ["unknown"]
        argument = self._bounded_argument_source(
            ctor_context, source.name)
        if argument is None:
            return ["unknown"]
        return self._bounded_source_candidates(
            ctor_context.caller_module, argument, ctor_context.parent,
            tracers, _seen)

    ## Evaluate a source under one exact local call context.
    #
    #  Parameter substitution is bounded by exact call positions. Nested local
    #  calls carry the current context as their parent, which supports simple
    #  forwarding without merging unrelated call sites.
    #
    #  @param source_module Module where source was evaluated.
    #  @param source Source value to resolve.
    #  @param context Current CallContext or None.
    #  @param tracers Dict of module name to analyzer.
    #  @param _seen Recursion guard.
    #  @return Candidate owner strings.
    def _bounded_source_candidates(self, source_module, source, context,
                                   tracers, _seen):
        source = normalize_source(source)
        context_key = (
            context.target.module,
            context.target.qualname,
            context.edge.call_lineno,
            context.edge.call_col_offset,
        ) if context is not None else ("", "", 0, 0)
        key = (
            "bounded-source", source_module, context_key,
            type(source).__name__, source_display(source),
        )
        if key in _seen:
            return ["unknown"]
        seen = set(_seen)
        seen.add(key)

        if isinstance(source, SourceSet):
            candidates = []
            for item in source.sources:
                candidates.extend(self._bounded_source_candidates(
                    source_module, item, context, tracers, set(seen)))
            return self._dedupe_list(candidates) or ["unknown"]
        if isinstance(source, UnknownSource):
            return ["unknown"]
        if (isinstance(source, DerivedResult)
                and source.kind == "receiver_preserving_ufunc"):
            candidate_groups = [
                self._bounded_source_candidates(
                    source_module, item, context, tracers, set(seen))
                for item in source.sources
            ]
            return [self._receiver_preserving_ufunc_owner(candidate_groups)]
        if (isinstance(source, DerivedResult)
                and source.kind == "expression"):
            candidate_groups = [
                (["python"] if self._returned_python_shape(
                    source_module, item, tracers, context) is not None else
                 self._bounded_source_candidates(
                     source_module, item, context, tracers, set(seen)))
                for item in source.sources
            ]
            return [self._bounded_expression_owner(candidate_groups)]
        if isinstance(source, ContainerItem):
            container = normalize_source(source.container)
            if (isinstance(container, CallResult)
                    and isinstance(source.index, int)):
                candidates = self._bounded_call_result_item_candidates(
                    source_module, container, source.index, tracers,
                    parent=context, _seen=seen)
                if candidates is not None:
                    return candidates
                return self._bounded_source_candidates(
                    source_module, container, context, tracers, seen)

        if isinstance(source, ContainerItem):
            container = normalize_source(source.container)
            if isinstance(container, ParameterSource):
                selected = self._bounded_pack_item_source(
                    context, container, source.index)
                if selected is None:
                    return ["unknown"]
                selected_module, selected_source, next_context = selected
                return self._bounded_source_candidates(
                    selected_module, selected_source, next_context,
                    tracers, seen)

        if isinstance(source, ParameterSource):
            current = context
            while current is not None:
                if source.scope == current.target.qualname:
                    argument = self._bounded_argument_source(
                        current, source.name)
                    if argument is None:
                        return ["unknown"]
                    return self._bounded_source_candidates(
                        current.caller_module, argument, current.parent,
                        tracers, seen)
                current = current.parent
            if context is not None:
                constructor_candidates = (
                    self._constructor_parameter_candidates(
                        context, source, tracers, seen))
                if constructor_candidates is not None:
                    return constructor_candidates
            # A local callee may return a parameter forwarded from its
            # enclosing project function.  No parent CallContext is attached
            # when this call expression is classified directly, so converge
            # that outer parameter over its collected project call edges.
            candidates = self._argument_owner_candidates(
                source_module, source, tracers)
            return candidates or ["unknown"]

        if (isinstance(source, str) and context is not None):
            current = context
            while current is not None:
                module_cg = self.project_cg.modules.get(
                    current.target.module)
                summary = (
                    module_cg.functions.get(current.target.qualname)
                    if module_cg is not None else None)
                if summary is not None and source in summary.params:
                    argument = self._bounded_argument_source(current, source)
                    if argument is None:
                        return ["unknown"]
                    return self._bounded_source_candidates(
                        current.caller_module, argument, current.parent,
                        tracers, seen)
                current = current.parent

        if isinstance(source, CallResult):
            nested = self._bounded_call_result_candidates(
                source_module, source, tracers,
                parent=context, _seen=seen)
            if nested is not None:
                return nested
            if source.result_source is not None:
                if (isinstance(source.result_source, str)
                        and source.result_source not in (
                            "", "local", "python", "unknown")):
                    return [source.result_source]
                return self._bounded_source_candidates(
                    source.source_module or source_module,
                    source.result_source, context, tracers, seen)
            if isinstance(source.callee, str):
                source_origin_module = source.source_module or source_module
                top = self._top_source(
                    source_origin_module, source.callee, tracers,
                    _seen=seen)
                return [top or "unknown"]
            return ["unknown"]

        return self._origin_candidates(
            source_module, source, tracers, _seen=seen)

    ## Resolve one local call result with exact call-site substitutions.
    #
    #  @param caller_module Module containing the call.
    #  @param source CallResult to resolve.
    #  @param tracers Dict of module name to analyzer.
    #  @param parent Enclosing forwarding context.
    #  @param _seen Recursion guard.
    #  @return Candidate owners, or None when no unique local context exists.
    def _bounded_call_result_candidates(self, caller_module, source, tracers,
                                        parent=None, _seen=None):
        if not isinstance(source, CallResult):
            return None
        contexts = self._bounded_call_contexts(
            caller_module, source.call_lineno, source.call_col_offset,
            tracers, parent=parent)
        if not contexts:
            return None
        seen = set(_seen or set())
        candidates = []
        for context in contexts:
            module_cg = self.project_cg.modules.get(context.target.module)
            summary = (
                module_cg.functions.get(context.target.qualname)
                if module_cg is not None else None)
            if summary is None or summary.returns is None:
                if len(contexts) == 1:
                    return None
                candidates.append("unknown")
                continue
            key = (
                "bounded-call", caller_module, source.call_lineno,
                source.call_col_offset, context.target.module,
                context.target.qualname,
            )
            if key in seen:
                candidates.append("unknown")
                continue
            context_seen = set(seen)
            context_seen.add(key)
            # A tuple return has no owner for the aggregate object. It is only
            # meaningful after the caller binds a proven positional item into
            # CallResult.result_source. If no positional marker is available,
            # evaluate every tuple item and retain only a common owner.
            if self._is_tuple_return_source(summary.returns):
                if source.result_source is not None:
                    candidates.extend(self._bounded_source_candidates(
                        context.target.module, source.result_source, context,
                        tracers, context_seen))
                    continue
                tuple_source = normalize_source(summary.returns)
                tuple_lengths = []
                branches = (tuple_source.sources
                            if isinstance(tuple_source, SourceSet)
                            else (tuple_source,))
                for branch in branches:
                    if isinstance(branch, DerivedResult):
                        tuple_lengths.append(len(branch.sources))
                if not tuple_lengths:
                    candidates.append("unknown")
                    continue
                for index in range(min(tuple_lengths)):
                    selected = self._return_item_source(
                        summary.returns, index)
                    if selected is None:
                        candidates.append("unknown")
                        break
                    candidates.extend(self._bounded_source_candidates(
                        context.target.module, selected, context,
                        tracers, set(context_seen)))
                continue
            candidates.extend(self._bounded_source_candidates(
                context.target.module, summary.returns, context,
                tracers, context_seen))
        return self._dedupe_list(candidates) or ["unknown"]

    ## Resolve one positional item from an exact local tuple-return call.
    #
    #  @param caller_module Module containing the call.
    #  @param source CallResult for the tuple-producing local call.
    #  @param index Zero-based selected tuple position.
    #  @param tracers Dict of module name to analyzer.
    #  @param parent Enclosing forwarding context.
    #  @param _seen Recursion guard.
    #  @return Candidate owners, or None when no tuple context is proven.
    def _bounded_call_result_item_candidates(
            self, caller_module, source, index, tracers,
            parent=None, _seen=None):
        if not isinstance(source, CallResult) or not isinstance(index, int):
            return None
        context = self._bounded_call_context(
            caller_module, source.call_lineno, source.call_col_offset,
            tracers, parent=parent)
        if context is None:
            return None
        module_cg = self.project_cg.modules.get(context.target.module)
        summary = (
            module_cg.functions.get(context.target.qualname)
            if module_cg is not None else None)
        if summary is None or summary.returns is None:
            return None
        selected = self._return_item_source(summary.returns, index)
        if selected is None:
            return None
        seen = set(_seen or set())
        key = (
            "bounded-call-item", caller_module,
            source.call_lineno, source.call_col_offset,
            context.target.module, context.target.qualname, index,
        )
        if key in seen:
            return ["unknown"]
        seen.add(key)
        return self._bounded_source_candidates(
            context.target.module, selected, context, tracers, seen)

    ## Converge bounded candidates without guessing across owners.
    #
    #  @param candidates Candidate owner strings.
    #  @return One owner or "unknown".
    def _bounded_candidates_top(self, candidates):
        unique = self._dedupe_list([
            candidate for candidate in candidates
            if candidate not in (None, "")
        ])
        if not unique or "unknown" in unique or len(unique) != 1:
            return "unknown"
        return unique[0]

    ## Find a module containing import evidence for a bounded owner.
    #
    #  The owner itself remains the public result. This module is used only
    #  so the existing trace engine can validate that result against the
    #  import which produced it.
    #
    #  @param owner Converged owner name.
    #  @param default_module Calling module fallback.
    #  @param tracers Dict of module name to analyzer.
    #  @return Module name containing matching import evidence.
    def _bounded_owner_module(self, owner, default_module, tracers):
        if owner in ("local", "python", "unknown", "", None):
            return default_module
        default_tracer = tracers.get(default_module)
        if _is_import_origin(default_tracer, owner):
            return default_module
        for candidate_module, tracer in tracers.items():
            if _is_import_origin(tracer, owner):
                return candidate_module
        return default_module

    ## Collect arguments supplied to one local function parameter.
    #
    #  Combines the legacy same-file call-site table with project CallEdge
    #  facts. The latter preserves forward references and cross-file calls.
    #  Each physical call site is returned once.
    #
    #  @param module Defining module.
    #  @param scope_name Qualified function name.
    #  @param parameter Parameter name.
    #  @param param_index Positional parameter index.
    #  @param tracer Defining-module analyzer.
    #  @param tracers Dict of module name to analyzer.
    #  @param prefer_protocol_shape Prefer independently proven PythonShape
    #  evidence without changing ordinary call-target argument sources.
    #  @param prefer_iterable_elements Prefer independently preserved
    #  container element sources without treating the container as an item.
    #  @param receiver_class_filter Optional project-local runtime class
    #  required by a virtual-dispatch forwarding context.
    #  @return List of (caller module, argument source) tuples.
    def _parameter_call_arguments(self, module, scope_name, parameter,
                                  param_index, tracer, tracers,
                                  prefer_protocol_shape=False,
                                  prefer_iterable_elements=False,
                                  receiver_class_filter=None):
        found = []
        seen = set()
        bare_scope = scope_name.rsplit(".", 1)[-1]

        for index, source in enumerate(
                tracer.parameter_sources.get((scope_name, parameter), [])):
            if source is None:
                continue
            key = ("parameterization", index, source_display(source))
            if key not in seen:
                seen.add(key)
                found.append((module, source))

        call_sites = []
        if receiver_class_filter is None:
            call_sites = (tracer.call_sites.get(scope_name)
                          or tracer.call_sites.get(bare_scope, []))
        for call_site in call_sites:
            if prefer_iterable_elements:
                continue
            args = call_site.get("args", [])
            if prefer_protocol_shape:
                protocol_args = call_site.get("protocol_args", [])
                if (param_index < len(protocol_args)
                        and protocol_args[param_index] is not None):
                    args = protocol_args
            if param_index >= len(args):
                continue
            arg_source = args[param_index]
            if arg_source is None:
                continue
            caller_module = call_site.get("module") or module
            key = (
                caller_module,
                call_site.get("lineno", 0),
                call_site.get("col_offset", 0),
            )
            if key not in seen:
                seen.add(key)
                found.append((caller_module, arg_source))

        ## A bounded callback edge can supply one exact iterable element to a
        # local callback parameter.  The only callback contract handled here
        # is multiprocessing.Pool.map(callback, iterable); arbitrary
        # external dispatch remains unresolved.
        if prefer_iterable_elements:
            target_name = scope_name.rsplit(".", 1)[-1]
            for caller_module, caller_cg in (
                    getattr(self, "project_cg", ProjectCallGraph()).modules.items()):
                caller_tracer = tracers.get(caller_module)
                for edge in caller_cg.edges:
                    if not self._is_multiprocessing_map_edge(
                            edge, caller_tracer):
                        continue
                    callback_positions = [
                        position for position, callback_name
                        in getattr(edge, "callback_args", {}).items()
                        if callback_name == target_name
                    ]
                    if len(callback_positions) != 1:
                        continue
                    iterable_position = callback_positions[0] + 1
                    element_source = getattr(
                        edge, "iterable_arg_sources", {}).get(
                            "pos", {}).get(iterable_position)
                    if element_source is None or isinstance(
                            element_source, UnknownSource):
                        continue
                    key = (caller_module, edge.call_lineno,
                           edge.call_col_offset, source_display(element_source))
                    if key not in seen:
                        seen.add(key)
                        found.append((caller_module, element_source))

        ## A bounded Process(target=..., args=(...)) edge supplies direct
        #  positional arguments to the named local callback.  This is kept
        #  separate from ordinary call-target matching because Process itself
        #  is the external callable, not the worker function.
        target_name = scope_name.rsplit(".", 1)[-1]
        for caller_module, caller_cg in (
                getattr(self, "project_cg", ProjectCallGraph()).modules.items()):
            caller_tracer = tracers.get(caller_module)
            for edge in caller_cg.edges:
                if not self._is_multiprocessing_process_edge(
                        edge, caller_tracer):
                    continue
                for binding in getattr(edge, "callback_bindings", []):
                    if binding.get("callback") != target_name:
                        continue
                    target_module_cg = self.project_cg.modules.get(module)
                    if (target_module_cg is None
                            or target_name not in target_module_cg.functions):
                        continue
                    callback_args = normalize_source(binding.get("args"))
                    if not isinstance(callback_args, TupleSource):
                        continue
                    if param_index >= len(callback_args.items):
                        continue
                    arg_source = callback_args.items[param_index]
                    key = (caller_module, edge.call_lineno,
                           edge.call_col_offset, source_display(arg_source))
                    if key not in seen:
                        seen.add(key)
                        found.append((caller_module, arg_source))

        cg = getattr(self, "project_cg", None)
        if cg is None:
            return found
        for caller_module, module_cg in cg.modules.items():
            caller_tracer = tracers.get(caller_module)
            for edge in module_cg.edges:
                if not self._edge_targets_local_function(
                        edge, caller_module, module, scope_name,
                        caller_tracer, tracers,
                        allow_inherited_dispatch=True):
                    continue
                if (receiver_class_filter is not None
                        and not self._edge_receiver_may_have_class(
                            edge, caller_module, receiver_class_filter,
                            tracers)):
                    continue
                target_cg = cg.modules.get(module)
                target_summary = (
                    target_cg.functions.get(scope_name)
                    if target_cg is not None else None)
                if target_summary is None:
                    target_summary = (
                        target_cg.functions.get(
                            scope_name.rsplit(".", 1)[-1])
                        if target_cg is not None else None)
                edge_args = self._edge_parameter_sources(
                    edge, target_summary, parameter, param_index,
                    prefer_protocol_shape=prefer_protocol_shape,
                    prefer_iterable_elements=prefer_iterable_elements)
                if edge_args is None:
                    continue
                for arg_source in edge_args:
                    key = (
                        caller_module,
                        edge.call_lineno,
                        edge.call_col_offset,
                        source_display(arg_source),
                    )
                    if key not in seen:
                        seen.add(key)
                        found.append((caller_module, arg_source))
        return found

    ## Check the narrow standard-library callback contract supported above.
    #  @param edge Candidate call edge.
    #  @return True only for multiprocessing.Pool.map(...).
    def _is_multiprocessing_map_edge(self, edge, tracer=None):
        if (not isinstance(getattr(edge, "callee_name", None), str)
                or not edge.callee_name.endswith(".map")):
            return False
        receiver = normalize_source(getattr(edge, "receiver_source", None))
        if not isinstance(receiver, CallResult):
            return False
        if receiver.callee == "multiprocessing":
            return True
        if tracer is None or not isinstance(receiver.callee, str):
            return False
        root = receiver.callee.split(".", 1)[0]
        return normalize_source(
            tracer.symbols.direct.get(root)) == "multiprocessing"

    ## Check the narrow standard-library Process callback contract.
    #  @param edge Candidate call edge.
    #  @param tracer Caller-module analyzer, used for direct imports.
    #  @return True only for multiprocessing.Process.
    def _is_multiprocessing_process_edge(self, edge, tracer=None):
        callee_name = getattr(edge, "callee_name", None)
        if callee_name == "multiprocessing.Process":
            return True
        if callee_name != "Process" or tracer is None:
            return False
        return (getattr(tracer, "import_from_symbols", {}).get("Process")
                == "multiprocessing.Process")

    ## Select sources supplied to one parameter from a resolved call edge.
    #  @param edge Call graph edge.
    #  @param summary Callee function signature summary.
    #  @param parameter Target parameter name.
    #  @param param_index Position in the flattened parameter list.
    #  @param prefer_protocol_shape Use independent PythonShape evidence.
    #  @param prefer_iterable_elements Use preserved iterable element evidence.
    #  @return List of sources, or None when the edge cannot bind safely.
    def _edge_parameter_sources(self, edge, summary, parameter, param_index,
                                prefer_protocol_shape=False,
                                prefer_iterable_elements=False):
        if summary is None:
            return None
        ordinary_args = edge.arg_sources
        if prefer_iterable_elements:
            ordinary_args = getattr(edge, "iterable_arg_sources", {})
        elif prefer_protocol_shape:
            ordinary_args = {
                "pos": dict(edge.arg_sources.get("pos", {})),
                "kw": dict(edge.arg_sources.get("kw", {})),
            }
            protocol_args = getattr(edge, "protocol_arg_sources", {})
            ordinary_args["pos"].update(protocol_args.get("pos", {}))
            ordinary_args["kw"].update(protocol_args.get("kw", {}))
        keyword_args = ordinary_args.get("kw", {})
        if parameter in keyword_args:
            return [keyword_args[parameter]]
        positional_params = list(
            getattr(summary, "positional_params", []))
        if not positional_params:
            positional_params = [
                name for name in summary.params
                if name not in (summary.vararg, summary.kwarg)
            ]
        if parameter == summary.vararg:
            start = len(positional_params)
            positional = ordinary_args.get("pos", {})
            values = [
                positional[index]
                for index in sorted(positional)
                if index >= start
            ]
            values.extend(getattr(edge, "star_arg_sources", {}).values())
            return values or None
        if parameter == summary.kwarg:
            explicit = set(positional_params)
            explicit.update(getattr(summary, "keyword_only_params", []))
            values = [
                value for name, value in keyword_args.items()
                if name not in explicit
            ]
            values.extend(getattr(edge, "star_kwarg_sources", []))
            return values or None
        if parameter in positional_params:
            index = positional_params.index(parameter)
            positional = ordinary_args.get("pos", {})
            if index in positional:
                return [positional[index]]
            star_item = self._star_positional_item_source(
                edge, summary, index)
            if star_item is not None:
                return [star_item]
        star_kwargs = getattr(edge, "star_kwarg_sources", [])
        if len(star_kwargs) == 1:
            return [ContainerItem(star_kwargs[0], parameter)]
        defaults = getattr(summary, "defaults", {})
        if parameter in defaults:
            return [defaults[parameter]]
        return None

    ## Collect the selected item of a variadic parameter from each exact
    #  project-local call edge.
    #  @param module Defining module of the variadic parameter.
    #  @param pack_source ParameterSource naming *args or **kwargs.
    #  @param index Integer position or keyword name.
    #  @param tracers Per-module analyzers.
    #  @return List of (caller module, selected source) tuples.
    def _parameter_pack_item_arguments(self, module, pack_source, index,
                                       tracers):
        if not isinstance(pack_source, ParameterSource):
            return []
        cg = getattr(self, "project_cg", None)
        target_cg = cg.modules.get(module) if cg is not None else None
        if target_cg is None:
            return []
        summary = target_cg.functions.get(pack_source.scope)
        if summary is None:
            summary = target_cg.functions.get(
                pack_source.scope.rsplit(".", 1)[-1])
        if summary is None or pack_source.name not in summary.params:
            return []
        param_index = summary.params.index(pack_source.name)
        found = []
        seen = set()
        for caller_module, caller_cg in cg.modules.items():
            caller_tracer = tracers.get(caller_module)
            for edge in caller_cg.edges:
                if not self._edge_targets_local_function(
                        edge, caller_module, module, pack_source.scope,
                        caller_tracer, tracers):
                    continue
                selected = self._edge_pack_item_source(
                    edge, summary, pack_source.name, index)
                if selected is None:
                    continue
                key = (caller_module, edge.call_lineno,
                       edge.call_col_offset, source_display(selected))
                if key in seen:
                    continue
                seen.add(key)
                found.append((caller_module, selected))
        return found

    ## Select one item from a variadic parameter on one call edge.
    #  @param edge Call graph edge.
    #  @param summary Callee signature summary.
    #  @param parameter Variadic parameter name.
    #  @param index Integer position or keyword name.
    #  @return Selected source or None when the edge is ambiguous.
    def _edge_pack_item_source(self, edge, summary, parameter, index):
        if parameter == summary.vararg:
            if not isinstance(index, int):
                return None
            positional_params = list(
                getattr(summary, "positional_params", []))
            actual_index = len(positional_params) + index
            positional = edge.arg_sources.get("pos", {})
            if actual_index in positional:
                return positional[actual_index]
            star_item = self._star_positional_item_source(
                edge, summary, actual_index)
            return star_item
        if parameter == summary.kwarg:
            if not isinstance(index, str):
                return None
            keyword_args = edge.arg_sources.get("kw", {})
            if index in keyword_args:
                positional_params = set(
                    getattr(summary, "positional_params", []))
                keyword_only = set(
                    getattr(summary, "keyword_only_params", []))
                if index not in positional_params | keyword_only:
                    return keyword_args[index]
            star_kwargs = getattr(edge, "star_kwarg_sources", [])
            if len(star_kwargs) == 1:
                return ContainerItem(star_kwargs[0], index)
        return None

    ## Return whether a CallEdge targets one specific local callable.
    #
    #  Handles module functions, class constructors, local methods, and
    #  statically enumerated callback-table values. Ambiguous local method
    #  receivers are accepted only as possible local targets; owner
    #  convergence still happens across every collected argument source.
    #
    #  @param edge Project call-graph edge.
    #  @param caller_module Module containing the call.
    #  @param target_module Defining module.
    #  @param scope_name Qualified target function name.
    #  @param caller_tracer Analyzer for caller module.
    #  @param tracers Dict of module name to analyzer.
    #  @param allow_inherited_dispatch Match inherited and overridden methods
    #  for parameter-flow analysis.
    #  @return True when the edge resolves to the target function.
    def _edge_targets_local_function(self, edge, caller_module,
                                     target_module, scope_name,
                                     caller_tracer, tracers,
                                     allow_inherited_dispatch=False):
        mapping_targets = getattr(edge, "mapping_targets", None)
        if mapping_targets is not None:
            return FunctionId(target_module, scope_name) in mapping_targets
        cg = getattr(self, "project_cg", None)
        target_cg = cg.modules.get(target_module) if cg is not None else None
        if target_cg is None:
            return False
        parts = scope_name.split(".")
        class_name = parts[-2] if len(parts) >= 2 else ""
        target_name = parts[-1]
        is_constructor = (
            target_name == "__init__"
            and class_name in target_cg.classes)
        is_method = (
            not is_constructor
            and class_name in target_cg.classes
            and target_name in target_cg.classes[class_name].methods)
        if len(parts) > 1 and not (is_constructor or is_method):
            return (caller_module == target_module
                    and normalize_source(edge.callee_source) == scope_name)

        callable_name = class_name if is_constructor else target_name
        if not is_method:
            callback_sources = normalize_source(
                getattr(edge, "callee_source", None))
            if isinstance(callback_sources, SourceSet):
                callback_values = callback_sources.sources
            else:
                callback_values = [callback_sources]
            for candidate in callback_values:
                candidate = normalize_source(candidate)
                if (isinstance(candidate, str)
                        and candidate.rsplit(".", 1)[-1] == callable_name
                        and (not is_constructor
                             or not isinstance(
                                 normalize_source(edge.callee),
                                 InstanceMethod))
                        and (not is_constructor or class_name == callable_name)):
                    return True

        # A local instance call such as ``f(value)`` is represented by the
        # variable name rather than ``__call__`` in the AST.  When the call
        # edge retains constructor sources for that variable, use those local
        # class identities to connect the edge to the corresponding
        # ``__call__`` method.  No class or library name is inferred from the
        # spelling of the variable itself.
        if is_method and target_name == "__call__":
            callable_classes = self._local_callable_class_candidates(
                caller_module,
                getattr(edge, "callee", None),
                caller_tracer,
                tracers,
                display_name=getattr(edge, "callee_name", ""),
            )
            for candidate_module, candidate_class in callable_classes:
                if self._local_class_is_or_derives(
                        candidate_module, candidate_class,
                        target_module, class_name, tracers):
                    return True

        callee_name = getattr(edge, "callee_name", "") or ""
        if not callee_name:
            return False
        if callee_name.rsplit(".", 1)[-1] != callable_name:
            return False

        if is_method:
            callee = normalize_source(edge.callee)
            if (edge.receiver_source is None
                    and not isinstance(callee, InstanceMethod)):
                # A bare module-function call cannot dispatch to a class
                # method just because the final identifier is the same.
                return False
            if (isinstance(callee, InstanceMethod)
                    and isinstance(callee.receiver, str)
                    and callee.receiver in target_cg.classes):
                receiver_classes = self._local_class_candidates(
                    caller_module, edge.receiver_source, tracers)
                if not receiver_classes:
                    local_identity = self._local_class_from_source(
                        caller_module, callee.receiver)
                    if local_identity is not None:
                        receiver_classes = [local_identity]
                if receiver_classes:
                    if not allow_inherited_dispatch:
                        return (target_module, class_name) in receiver_classes
                    if edge.receiver_source == "self":
                        return any(
                            self._local_class_is_or_derives(
                                target_module, class_name,
                                candidate_module, candidate_class, tracers)
                            for candidate_module, candidate_class
                            in receiver_classes
                        )
                    return any(
                        self._local_class_is_or_derives(
                            candidate_module, candidate_class,
                            target_module, class_name, tracers)
                        for candidate_module, candidate_class
                        in receiver_classes
                    )
            receiver_class = self._resolve_local_class_identity(
                caller_module, edge.receiver_source, tracers)
            if receiver_class is not None:
                if allow_inherited_dispatch:
                    return self._local_class_is_or_derives(
                        receiver_class[0], receiver_class[1],
                        target_module, class_name, tracers)
                return receiver_class == (target_module, class_name)
            if edge.receiver_source == "self":
                caller_parts = edge.caller.qualname.rsplit(".", 1)
                caller_class = (
                    caller_parts[0] if len(caller_parts) == 2 else "")
                caller_cg = self.project_cg.modules.get(caller_module)
                if (caller_cg is not None
                        and caller_class in caller_cg.classes):
                    if allow_inherited_dispatch:
                        return self._local_class_is_or_derives(
                            target_module, class_name,
                            caller_module, caller_class, tracers)
                    return (
                        caller_module == target_module
                        and caller_class == class_name)
                return False
            if callee in ("local", "self"):
                return True
            receiver_candidates = self._argument_owner_candidates(
                caller_module, edge.receiver_source, tracers)
            return self._dedupe_list(receiver_candidates) == ["local"]

        if not is_constructor and edge.receiver_source is not None:
            receiver_class = self._resolve_local_class_identity(
                caller_module, edge.receiver_source, tracers)
            if receiver_class is not None or edge.receiver_source == "self":
                return False

        first = callee_name.split(".", 1)[0]
        if caller_tracer is not None:
            imported = caller_tracer.import_from_symbols.get(first)
            if imported:
                return imported == target_module + "." + callable_name
            direct = normalize_source(
                caller_tracer.symbols.direct.get(first))
            if isinstance(direct, str) and direct not in (
                    "", "local", "python", "unknown"):
                if direct == target_module:
                    return True
                if direct.endswith("." + callable_name):
                    return direct == target_module + "." + callable_name
                if not self.is_local(direct):
                    return False

        if caller_module == target_module and callee_name == callable_name:
            return True

        definitions = []
        if cg is not None:
            for candidate_module, candidate_cg in cg.modules.items():
                if is_constructor and callable_name in candidate_cg.classes:
                    definitions.append(candidate_module)
                elif (not is_constructor
                      and callable_name in candidate_cg.functions):
                    definitions.append(candidate_module)
        return len(definitions) == 1 and definitions[0] == target_module

    ## Unify receiver object ownership lookup through a single entry point.
    #
    #  Checks the receiver's provenance in symbols.direct and traces
    #  the callee / import alias to determine the owning library.
    #  Currently covers factory return tracing (A1) and import alias
    #  resolution (Case B).  Constructor provenance (A2) is gated on
    #  single-file callee naming and will activate once call_lookup
    #  returns the alias name rather than the module path.
    #
    #  @param module The current module.
    #  @param receiver The receiver variable name.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param tracers Dict of module_name → SingleFileAnalyzer.
    #  @return Top library name, or None.
    def _resolve_receiver_object_top(self, module, receiver, tracer, tracers):
        sd = normalize_source(tracer.symbols.direct.get(receiver))
        if sd is None:
            return None
        # Case A: CallResult — receiver was created by a call.
        # Try factory return tracing first, then constructor provenance.
        if isinstance(sd, CallResult) and isinstance(sd.callee, str):
            callee = sd.callee
            # A1: Factory return tracing (local functions with return_sources).
            top = self._resolve_receiver_with_return_sources(
                callee, module, tracers, set())
            if top and top not in ("local", "python", "unknown", ""):
                return top
            # A2: Constructor provenance.
            # The callee may be a simple name resolvable through symbols.direct
            # (e.g. from ext.api import Session; s = Session() → callee="Session").
            if isinstance(callee, str) and '.' not in callee:
                callee_sd = normalize_source(tracer.symbols.direct.get(callee))
                if isinstance(callee_sd, str) and callee_sd not in ("local", "python", ""):
                    top = self._top_source(module, callee_sd, tracers)
                    if top and top not in ("local", "python", "unknown", ""):
                        return top
            return None
        # Case B: Import alias (e.g. from factory import create_app).
        if isinstance(sd, str) and sd not in ("local", "python", ""):
            return self._resolve_receiver_with_return_sources(
                receiver, module, tracers, set())
        return None

    ## Resolve a receiver name through cross-file return_sources.
    #
    #  1.0.5 P1: supports app.test_client() → flask when app traces
    #  to a CallResult (local or imported factory function) whose
    #  return_sources trace to an import-backed library.
    #
    #  Handles:
    #    from factory import create_app → callee="create_app"
    #    import factory → callee="factory.create_app"
    #    import factory as f → callee="f.create_app"
    #
    #  @param callee The CallResult callee name (e.g. "create_app").
    #  @param module Current module.
    #  @param tracers Dict of module_name → SingleFileAnalyzer.
    #  @param _visited Already-visited set for cycle detection.
    #  @return Top library name or None.
    def _resolve_receiver_with_return_sources(self, callee, module, tracers, _visited):
        if not isinstance(callee, str):
            return None
        if (module, callee) in _visited:
            return None
        _visited.add((module, callee))

        # Split dotted callee to find defining module and function name.
        # factory.create_app → target_module="factory", func="create_app"
        # pkg.factory.create_app → target_module="pkg.factory", func="create_app"
        parts = callee.split('.')
        func = parts[-1]
        target_mod = None
        for i in range(len(parts) - 1, 0, -1):
            candidate_mod = '.'.join(parts[:i])
            if candidate_mod in tracers:
                target_mod = candidate_mod
                break

        if target_mod is None:
            # Simple name: check import_from_symbols first so aliased
            # imports resolve to the real function name.
            # from factory import make_session as cross_make
            #   → import_from_symbols["cross_make"] = "factory.make_session"
            tracer = tracers.get(module)
            if tracer is not None:
                imported = getattr(tracer, "import_from_symbols", {}).get(callee)
                if imported:
                    imported_parts = imported.split(".")
                    for i in range(len(imported_parts) - 1, 0, -1):
                        candidate_mod = ".".join(imported_parts[:i])
                        if candidate_mod in tracers:
                            target_mod = candidate_mod
                            func = ".".join(imported_parts[i:])
                            break
            if target_mod is None and tracer is not None:
                sd = tracer.symbols.direct.get(callee)
                if isinstance(sd, str) and sd in tracers:
                    target_mod = sd

        if target_mod is None:
            target_mod = module

        target_tracer = tracers.get(target_mod)
        if target_tracer is None:
            return None

        # Check return_sources for the function in the defining module
        rs = target_tracer.return_sources.get(func)
        if rs is not None:
            rs_norm = normalize_source(rs)
            sources = rs_norm.sources if isinstance(rs_norm, SourceSet) else [rs_norm]
            for s in sources:
                s = normalize_source(s)
                if isinstance(s, CallResult) and isinstance(s.callee, str):
                    top = self._top_source(target_mod, s.callee, tracers)
                    if top and top not in ("local", "python", "unknown", ""):
                        return top
        return None

    ## Try to resolve a local class method to an external source.
    #
    #  Checks whether the method's return_sources trace to a constructor
    #  parameter that has external provenance via call-site arguments.
    #  @param module The current module.
    #  @param class_name The local class name.
    #  @param method_name The method being called.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return External library name, or None.
    def _resolve_local_method_to_external(self, module, class_name,
                                           method_name, receiver,
                                           tracer, tracers):
        ## Check qualname first so class methods don't share bare keys.
        qkey = class_name + "." + method_name
        rs = tracer.return_sources.get(qkey)
        if not rs:
            rs = tracer.return_sources.get(method_name)
        if not rs:
            return None
        rs = normalize_source(rs)
        sources = rs.sources if isinstance(rs, SourceSet) else [rs]
        for src in sources:
            if isinstance(src, InstanceMethod):
                param_name = src.receiver
                if isinstance(param_name, str):
                    ctor_key = class_name + ".__init__"
                    ctor_params = (tracer.function_params.get("__init__", [])
                                   or tracer.function_params.get(ctor_key, []))
                    if param_name in ctor_params:
                        param_idx = ctor_params.index(param_name)
                        call_sites = (tracer.call_sites.get("__init__", [])
                                      or tracer.call_sites.get(ctor_key, []))
                        match_ln, match_col = self._receiver_ctor_pos(
                            receiver, tracer)
                        matched = None
                        for cs in call_sites:
                            if param_idx >= len(cs.get("args", [])):
                                continue
                            cs_ln = cs.get("lineno", 0)
                            cs_col = cs.get("col_offset", 0)
                            if (match_ln and cs_ln == match_ln
                                    and cs_col == match_col):
                                matched = cs
                                break
                            if not match_ln:
                                matched = cs
                        if matched:
                            arg_src = matched["args"][param_idx]
                            arg_src = normalize_source(arg_src)
                            if isinstance(arg_src, CallResult):
                                top = self._top_source(
                                    module, arg_src.callee, tracers)
                                if top and top not in ("local", "python",
                                                       "unknown", ""):
                                    return top
                            if isinstance(arg_src, str):
                                top = self._top_source(module, arg_src, tracers)
                                if top and top not in ("local", "python",
                                                       "unknown", ""):
                                    return top
        return None

    ## Get the constructor call-site position for a receiver instance.
    #  @param receiver The variable name bound to a class instance.
    #  @param tracer The SingleFileAnalyzer for the module.
    #  @return Tuple of (lineno, col_offset) or (0, 0) if unknown.
    def _receiver_ctor_pos(self, receiver, tracer):
        if not receiver or not isinstance(receiver, str):
            return (0, 0)
        sd = tracer.symbols.direct.get(receiver)
        sd = normalize_source(sd)
        if isinstance(sd, CallResult):
            return (sd.call_lineno, sd.call_col_offset)
        return (0, 0)

    ## Resolve a parameter name to its call-site argument for a specific callee.
    #
    #  Unlike _trace_parameter_source, this only searches the given callee's
    #  call-sites, preventing false positives from same-named parameters in
    #  other functions.
    ## Look up the return source of a local function via call-graph facts.
    #
    #  Searches the current module's ModuleCallGraph first, then falls back
    #  to all modules (for cross-file imported local functions).  Returns a
    #  single source only when unambiguous; SourceSet with multiple
    #  import-backed sources is left for the classifier (7B-full PR2).
    #  @param cur_module The module where the call occurs.
    #  @param callee Bare function name or qualname (e.g. "make_array").
    #  @return A non-local source string, or None.
    def _lookup_cg_return_source(self, cur_module, callee):
        if not isinstance(callee, str):
            return None
        cg = getattr(self, 'project_cg', None)
        if cg is None:
            return None
        # Search current module first.  Only fall back to other modules
        # when the function is NOT found locally (cross-file import case).
        search_modules = list(cg.modules.keys())
        if cur_module and cur_module in search_modules:
            search_modules.remove(cur_module)
            search_modules.insert(0, cur_module)
        for module in search_modules:
            mcg = cg.modules.get(module)
            if mcg is None:
                continue
            fs = mcg.functions.get(callee)
            if fs is None:
                continue
            # Found in this module — if no returns, don't fall back to others.
            if fs.returns is None:
                return None
            # Tuple return summaries require caller-side positional binding;
            # they are not whole-result sources for the legacy resolver.
            if self._is_tuple_return_source(fs.returns):
                return None
            returns_norm = normalize_source(fs.returns)
            if isinstance(returns_norm, SourceSet):
                # Collect import-backed tops; return single if unambiguous.
                tops = []
                for src in returns_norm.sources:
                    src_norm = normalize_source(src)
                    if isinstance(src_norm, str) and not self.is_local(src_norm):
                        top = self._top_name(src_norm)
                        if top and top not in ("local", "unknown", ""):
                            tops.append(top)
                    elif isinstance(src_norm, InstanceMethod) and isinstance(src_norm.receiver, str):
                        top = self._top_name(src_norm.receiver)
                        if top and top not in ("local", "unknown", ""):
                            tops.append(top)
                    elif isinstance(src_norm, CallResult) and isinstance(src_norm.callee, str):
                        top = self._top_name(src_norm.callee)
                        if top and top not in ("local", "unknown", ""):
                            tops.append(top)
                if len(tops) == 1:
                    return tops[0]
                # Multiple or zero import-backed sources — let classifier handle.
                continue
            if isinstance(returns_norm, str) and not self.is_local(returns_norm):
                top = self._top_name(returns_norm)
                if top and top not in ("local", "unknown", ""):
                    return top
            elif isinstance(returns_norm, CallResult):
                if isinstance(returns_norm.callee, str):
                    top = self._top_name(returns_norm.callee)
                    if top and top not in ("local", "unknown", ""):
                        return top
        return None

    # _is_container_receiver and _lookup_cg_edge_arg_source removed
    # (1.0.5 P0 cleanup).  Arg-source evidence belongs in
    # SymbolProvenance, not ApiCall.top_library.

    ## Look up import-backed constructor attr used by a specific method (7B-full PR3).
    #
    #  Searches ProjectCallGraph for edges where the method (identified by
    #  class_name + method_name) calls through a self.attr whose constructor
    #  source is import-backed.  Only attrs actually used by the method are
    #  considered — a class's unrelated import-backed attrs do not leak.
    #  @param cur_module The module where the class is defined.
    #  @param class_name The local class name.
    #  @param method_name The method being called (e.g. "fit").
    #  @return Tuple of (src_module, top_library) or None.
    def _lookup_cg_class_attr_source(self, cur_module, class_name, method_name):
        if not isinstance(class_name, str) or not isinstance(method_name, str):
            return None
        cg = getattr(self, 'project_cg', None)
        if cg is None:
            return None
        # Search current module first, then all modules.
        search_modules = list(cg.modules.keys())
        if cur_module and cur_module in search_modules:
            search_modules.remove(cur_module)
            search_modules.insert(0, cur_module)
        for module in search_modules:
            mcg = cg.modules.get(module)
            if mcg is None:
                continue
            cs = mcg.classes.get(class_name)
            if cs is None:
                continue
            # Collect import-backed attrs with their library provenance.
            import_attrs = {}  # attr_name -> (src_module, top)
            for attr_name, src in cs.attrs.items():
                src_norm = normalize_source(src)
                if isinstance(src_norm, CallResult):
                    if isinstance(src_norm.callee, str):
                        callee = src_norm.callee
                        if _is_builtin(callee) or self.is_local(callee):
                            continue
                        top = self._top_name(callee)
                        if top and top not in ("local", "unknown", ""):
                            import_attrs[attr_name] = (module, top)
                elif isinstance(src_norm, str):
                    if (self.is_local(src_norm) or src_norm in ("local", "unknown", "")
                            or _is_builtin(src_norm)):
                        continue
                    bare = attr_name[5:] if attr_name.startswith("self.") else attr_name
                    if src_norm == bare or src_norm.startswith(bare):
                        continue
                    if '.' not in src_norm:
                        continue
                    top = self._top_name(src_norm)
                    if top and top not in ("local", "unknown", ""):
                        import_attrs[attr_name] = (module, top)
            if not import_attrs:
                return None
            # Only return an attr if the method actually uses it.
            # Check edges where caller is this method.
            method_qualname = class_name + "." + method_name
            method_tops = {}  # top -> (src_module, top)
            for edge in mcg.edges:
                if edge.caller.qualname != method_qualname:
                    continue
                rcvr = edge.receiver_source
                if rcvr is None:
                    continue
                rcvr_norm = normalize_source(rcvr)
                # Match receiver against import-backed attrs.
                for attr_name, (attr_mod, attr_top) in import_attrs.items():
                    if self._edge_receiver_matches_attr(rcvr_norm, cs, attr_name):
                        method_tops[attr_top] = (attr_mod, attr_top)
            if len(method_tops) == 1:
                return list(method_tops.values())[0]
            # Multiple candidates — return None, let classifier handle alternatives.
            return None
        return None

    ## Check whether a CallEdge receiver matches a ClassSummary attr source.
    #  @param rcvr_norm Normalized receiver source from the edge.
    #  @param cs The ClassSummary.
    #  @param attr_name The attr name to check (e.g. "self.gp").
    #  @return True if the receiver matches the attr's source.
    def _edge_receiver_matches_attr(self, rcvr_norm, cs, attr_name):
        attr_src = cs.attrs.get(attr_name)
        if attr_src is None:
            return False
        attr_norm = normalize_source(attr_src)
        if isinstance(rcvr_norm, CallResult) and isinstance(attr_norm, CallResult):
            return rcvr_norm.callee == attr_norm.callee
        if isinstance(rcvr_norm, str) and isinstance(attr_norm, str):
            return rcvr_norm == attr_norm
        return rcvr_norm == attr_norm

    #  @param module The module where the call occurs.
    #  @param callee The function name whose parameter is being resolved.
    #  @param param_name The parameter name to resolve.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return A source value from the call-site argument, or None.
    def _resolve_param_to_arg(self, module, callee, param_name, tracers,
                               call_lineno=0, call_col_offset=0):
        tr = tracers.get(module)
        if not tr or not isinstance(param_name, str):
            return None
        params = tr.function_params.get(callee, [])
        if param_name not in params:
            return None
        param_idx = params.index(param_name)
        best = None
        for call_site in tr.call_sites.get(callee, []):
            if param_idx >= len(call_site["args"]):
                continue
            best = call_site["args"][param_idx]
            if call_lineno:
                cs_lineno = call_site.get("lineno", 0)
                cs_col = call_site.get("col_offset", 0)
                if cs_lineno == call_lineno and cs_col == call_col_offset:
                    return best
        return best

    ## Recursively trace a symbol through cross-file imports to its origin.
    #
    #  Follows direct sources across module boundaries, handling
    #  structured sources (container items, instance methods, container iters,
    #  call results) at each step.
    #  @param module The current module being traced from.
    #  @param symbol The symbol to trace.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @param visited Set of already-visited (module, symbol) pairs.
    #  @return Ordered chain list from symbol to origin.
    # ── trace engine ─────────────────────────────────────────────────────

    def trace_symbol(self, module, symbol, tracers, visited, _direct_source=None):
        if (module, symbol) in visited:
            return []
        visited.add((module, symbol))
        tracer = tracers.get(module)
        if not tracer:
            return []
        direct_source = _direct_source if _direct_source is not None else tracer.symbols.direct.get(symbol)
        if not direct_source:
            if isinstance(symbol, str) and '.' in symbol:
                prefix = symbol.split('.')[0]
                if prefix in tracer.symbols.direct:
                    sub_chain = self.trace_symbol(module, prefix, tracers, visited)
                    if sub_chain:
                        return [symbol] + sub_chain
            # A local wildcard import is only a fallback for unresolved names.
            # Preserve explicit external import evidence before considering it.
            if (isinstance(symbol, str)
                    and '.' in symbol
                    and _is_import_origin(tracer, symbol)):
                full_symbol = self.module_mapper.resolve_module_name(
                    symbol, module)
                if not self.is_local(full_symbol):
                    return [symbol]
            if tracer.wildcard_modules:
                tops = []
                local_found = False
                for wm in tracer.wildcard_modules:
                    actual_wm = wm
                    if wm not in tracers:
                        for m in tracers:
                            if m == wm or m.endswith('.' + wm):
                                actual_wm = m
                                break
                    if self.is_local(actual_wm):
                        local_found = True
                    else:
                        top = wm.split('.')[0]
                        if top not in tops:
                            tops.append(top)
                if tops:
                    if len(tops) == 1:
                        return [symbol, tops[0]]
                    else:
                        return [symbol, "[" + ",".join(tops) + "]"]
                if local_found:
                    for wm in tracer.wildcard_modules:
                        actual_wm = wm
                        if wm not in tracers:
                            for m in tracers:
                                if m == wm or m.endswith('.' + wm):
                                    actual_wm = m
                                    break
                        if self.is_local(actual_wm):
                            src_tracer = tracers.get(actual_wm)
                            if src_tracer and symbol in src_tracer.symbols.direct:
                                sub_chain = self.trace_symbol(actual_wm, symbol, tracers, visited)
                                if sub_chain:
                                    return [symbol] + sub_chain
                    return [symbol, "local"]
            if symbol == "self" or (isinstance(symbol, str) and symbol.startswith("self.")):
                return [symbol, "local"]
            if isinstance(symbol, str) and _is_builtin(symbol):
                return [symbol, "python"]
            return [symbol]

        if direct_source == "local":
            param_chain = self._trace_parameter_source(
                module, symbol, symbol, tracer, tracers, visited)
            if param_chain:
                return param_chain

        if isinstance(direct_source, str) and direct_source != symbol:
            param_chain = self._trace_parameter_source(
                module, direct_source, symbol, tracer, tracers, visited)
            if param_chain:
                return param_chain

        structured = self._resolve_structured_source(
            module, direct_source, tracers, _seen=set(visited))
        if structured is not None:
            display_name, src_module, src_symbol = structured
            sub_chain = self.trace_symbol(src_module, src_symbol, tracers, visited)
            if sub_chain and sub_chain != [src_symbol]:
                return [symbol, display_name] + sub_chain
            src_tracer = tracers.get(src_module)
            if (
                sub_chain == [src_symbol]
                and isinstance(src_symbol, str)
                and src_tracer is not None
                and _is_import_origin(src_tracer, src_symbol)
            ):
                return [symbol, display_name, src_symbol]
            if isinstance(src_symbol, str) and ('.' in src_symbol or '[' in src_symbol or src_symbol == 'local' or _is_builtin(src_symbol) or src_symbol in ("unknown", "python")):
                if '.' in src_symbol:
                    first = src_symbol.split('.')[0]
                    full_first = self.module_mapper.resolve_module_name(first, src_module)
                    if self.is_local(full_first):
                        return [symbol, display_name, src_module]
                return [symbol, display_name, src_symbol]
            ## 1.0.5 P2: explicit result_source (__import__("os") → "os")
            #  is already a terminal library name — use directly.
            if (isinstance(direct_source, CallResult)
                    and isinstance(getattr(direct_source, 'result_source', None), str)
                    and src_symbol == getattr(direct_source, 'result_source', None)):
                return [symbol, display_name, src_symbol]
            return [symbol, display_name, src_module]

        if isinstance(direct_source, tuple):
            return [symbol, str(direct_source)]

        if isinstance(direct_source, str):
            full_source = self.module_mapper.resolve_module_name(direct_source, module)
        else:
            full_source = direct_source

        if self.is_local(full_source):
            sub_chain = self.trace_symbol(full_source, symbol, tracers, visited)
            if sub_chain and sub_chain != [symbol]:
                return [symbol, full_source] + sub_chain
            else:
                return [symbol, full_source]
        elif isinstance(full_source, str) and full_source in tracer.symbols.direct:
            sub_chain = self.trace_symbol(module, full_source, tracers, visited)
            if sub_chain:
                return [symbol] + sub_chain
            else:
                return [symbol, full_source]
        else:
            return [symbol, full_source]

    ## Extract the top-level name from a dotted name string.
    #  @param name Possibly dotted name.
    #  @return The first component before a dot.
    def _top_name(self, name):
        if isinstance(name, str) and "." in name:
            return name.split(".")[0]
        return name

    ## Resolve a symbol to its top-level source library.
    #
    #  Traces through the chain and returns the top-level name, or "python"
    #  for builtins.
    #  @param src_module The module where the symbol is referenced.
    #  @param symbol The symbol to resolve.
    #  @param tracers Dict of module_name -> SingleFileAnalyzer.
    #  @return Top-level library name (e.g. "requests", "python").
    def _top_source(self, src_module, symbol, tracers, _seen=None):
        if not symbol:
            return None
        if isinstance(symbol, PythonShape):
            return "python"
        if isinstance(symbol, str) and _is_builtin(symbol):
            return "python"
        src_tracer = tracers.get(src_module)
        if src_tracer and self._is_known_local_symbol(src_tracer, symbol):
            return "local"
        visited = set(_seen) if _seen is not None else set()
        chain = self.trace_symbol(src_module, symbol, tracers, visited)
        if chain:
            top = self.extract_final_source(chain)
            if top in ("local", "python", "unknown", ""):
                return top
            if isinstance(symbol, str) and chain in ([symbol], [self._top_name(symbol)]):
                # Merged container candidates like "[requests,numpy]"
                # are not local/import-origin names — return as-is.
                if isinstance(symbol, str) and symbol.startswith("[") and symbol.endswith("]"):
                    return symbol
                if self.is_local(symbol):
                    return "local"
                if src_tracer and _is_import_origin(src_tracer, symbol):
                    return self._top_name(symbol)
                # 1.0.5 P1: cross-file return tracing may resolve to a
                # library name imported in another module (e.g. flask
                # from factory tracing).  Only for simple names that
                # are clearly not local sub-modules.
                if isinstance(symbol, str) and '.' not in symbol:
                    all_mods = self.module_mapper.get_all_modules()
                    if not any(m.endswith('.' + symbol) for m in all_mods):
                        for _mt, mt_tracer in tracers.items():
                            if _is_import_origin(mt_tracer, symbol):
                                return self._top_name(symbol)
                return "unknown"
            return top
        if src_tracer:
            top = src_tracer.symbols.get_top(symbol)
            if top:
                return self._top_name(top)
            if isinstance(symbol, str) and _is_import_origin(src_tracer, symbol):
                return self._top_name(symbol)
        return "unknown"

    ## Extract the ultimate source from a resolution chain.
    #
    #  Walks the chain in reverse; the first non-local, non-builtin element
    #  is the top-level library.
    #  @param chain The resolution chain list.
    #  @return Final source string.
    def extract_final_source(self, chain):
        if not chain:
            return ""
        found_local_module = False
        for item in reversed(chain):
            if isinstance(item, str) and _is_builtin(item):
                return "python"
            if isinstance(item, str) and not self.is_local(item):
                if found_local_module:
                    return "local"
                result = self._top_name(item)
                if result == "self":
                    return "local"
                return result
            if isinstance(item, str) and self.is_local(item):
                found_local_module = True
        return "local"


## Analyze an entire project and return structured results.
#
#  Convenience function: creates a ProjectAnalyzer, runs analysis, and
#  returns a ProjectAnalysis object.
#  @param project_root Absolute path to the project root directory.
#  @return ProjectAnalysis with all per-file and cross-file results.
## Analyze an entire project and return structured results.
#
#  Convenience function: creates a ProjectAnalyzer, runs analysis, and
#  returns a ProjectAnalysis object.
#  @param project_root Absolute path to the project root directory.
#  @return ProjectAnalysis with all per-file and cross-file results.
def analyze_project(project_root):
    analyzer = ProjectAnalyzer(project_root)
    return analyzer.analyze()
