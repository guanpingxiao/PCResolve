## @package pcresolve.single_file
#  Provide single-file AST-based API call tracing.
#
#  Contains the SingleFileAnalyzer class which visits every node in a
#  Python file's AST and builds a symbol table + list of API calls with
#  their resolved primary owners and supporting provenance.

import ast
import builtins
from dataclasses import replace
from .mapping_facts import MappingFacts, bound_names as mapping_bound_names
from .symbol_table import SymbolTable
from .ir import CallSite, SymbolRef
from .scope import (Scope, Binding, SCOPE_MODULE, SCOPE_FUNCTION, SCOPE_CLASS,
                       SCOPE_COMPREHENSION, merge_snapshots)
from .sources import (ContainerItem, ContainerIter, TupleSource, InstanceMethod,
                       ParameterSource, InstanceAttribute, PythonShape,
                       SuperMethod, CallResult,
                       DerivedResult, UnknownSource,
                       SourceSet, is_structured_source, normalize_source,
                       source_display, make_source_set)
from .call_graph import (FunctionId, FunctionSummary, ClassSummary, CallEdge,
                         IterationBinding, ModuleCallGraph)

## Python 2 builtins not present in Python 3's builtins module.
_PY2_BUILTINS = frozenset({
    "apply", "basestring", "buffer", "cmp", "coerce", "execfile",
    "file", "intern", "long", "raw_input", "reduce", "reload",
    "StandardError", "unichr", "unicode", "xrange",
})

## 1.0.5 P1: builtin container/type method names whose receiver is a
#  Python-provided object even when the receiver variable is local.
#  When a call like x.append(...) has a receiver tracing to "local"
#  and the method name is in this set, the callable owner is python.
## 1.0.5 P1: builtin container/type methods keyed by container kind.
#  The container kind (list/dict/set/tuple/str) provides context so
#  that method classification is safe from local-class name collisions.
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


## Check if a name is a Python builtin (including Python 2 builtins).
def _is_builtin(name):
    return isinstance(name, str) and (hasattr(builtins, name) or name in _PY2_BUILTINS)


## Return the runtime type corresponding to a proven PythonShape kind.
#  @param kind Concrete builtin type or container kind.
#  @return Builtin type object, or None for an unsupported shape.
def _builtin_shape_type(kind):
    if kind == "NoneType":
        return type(None)
    value = getattr(builtins, kind, None)
    return value if isinstance(value, type) else None


## Check whether a proven PythonShape exposes a callable attribute.
#  This consults the Python builtin type itself instead of maintaining a
#  method-name allowlist. Unknown receivers never reach this helper.
#  @param kind Concrete builtin type or container kind.
#  @param method Attribute name being called.
#  @return True when the builtin type defines a callable attribute.
def _has_builtin_shape_method(kind, method):
    shape_type = _builtin_shape_type(kind)
    return shape_type is not None and callable(getattr(shape_type, method, None))


## Return an established builtin method's result shape.
#  @param receiver Independently proven PythonShape, not an owner name.
#  @param method Called attribute name.
#  @return PythonShape or None without a result protocol.
def _builtin_method_return_shape(receiver, method):
    if not isinstance(receiver, PythonShape):
        return None
    if receiver.kind == "str":
        if method in ("split", "rsplit", "splitlines"):
            return PythonShape("list", "str")
        if method in (
                "strip", "rstrip", "lstrip", "replace", "upper", "lower",
                "title", "capitalize", "swapcase", "center", "ljust",
                "rjust", "zfill", "format", "format_map", "join"):
            return PythonShape("str")
    if method == "copy" and receiver.kind in ("list", "dict", "set"):
        return receiver
    return None


## 1.0.5 P2: builtin return-object ownership semantics.
#
#  Maps builtin callable names to their result_source:
#  - "python": result object is a Python-provided type (open, list, str, …).
#  - None: use legacy return_sources / call-graph resolution.
#  - "unknown": result is statically unresolvable (eval, dynamic __import__).
_BUILTIN_PYTHON_OWNED_RESULT = frozenset({
    "open", "super",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "bytes", "bytearray", "complex", "frozenset", "object",
    "range", "slice", "memoryview",
    "staticmethod", "classmethod", "property",
    "enumerate", "filter", "map", "zip", "sorted",
    "len", "print", "exec",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "RuntimeError", "StopIteration", "OSError", "NotImplementedError",
    "AttributeError", "ImportError", "NameError", "SyntaxError",
    "ZeroDivisionError", "OverflowError", "EOFError", "IOError",
    "FileNotFoundError", "StopAsyncIteration",
})

_BUILTIN_ARBITRARY_RESULT = frozenset({
    "eval", "exec",
})

_BUILTIN_ELEMENT_DERIVED = frozenset({"next", "min", "max"})
_BUILTIN_PROTOCOL_DERIVED = frozenset({"abs"})


## Check whether a builtin name is not shadowed by a local definition.
#
#  @param self SingleFileAnalyzer instance.
#  @param node The ast.Call node.
#  @return True if the call is to an unshadowed builtin.
def _is_unshadowed_builtin_call(tracer, node):
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name):
        return False
    name = node.func.id
    if not _is_builtin(name):
        return False
    if name in tracer.defined_functions:
        return False
    if name in tracer.import_from_symbols:
        return False
    if name in tracer.local:
        return False
    # Check lexical scope binding for shadowing.
    binding = tracer.current_scope().lookup(
        name, skip_parent_classes=True)
    if binding is not None:
        return False
    return True


## Trace an expression used as a builtin result candidate.
#
#  Literal values are Python-owned.  Other expressions retain their full
#  source IR so cross-file resolution can preserve every possible owner.
#  @param node Candidate AST expression.
#  @param trace_fn Callable to trace an AST expression.
#  @return Source value or UnknownSource.
def _builtin_value_source(node, trace_fn):
    if isinstance(node, ast.Constant):
        return "python"
    source = trace_fn(node)
    if source is not None:
        return source
    return UnknownSource("builtin value")


## Resolve the element source carried by an iterable expression.
#
#  Named containers use ContainerIter so their tracked item bindings remain
#  available cross-file.  iter()/reversed() results expose their element
#  evidence without claiming that the iterator object's owner is the owner of
#  each yielded value.
#  @param node Iterable AST expression.
#  @param trace_fn Callable to trace an AST expression.
#  @return Source value describing the iterable's possible elements.
def _iterable_element_source(node, trace_fn):
    if isinstance(node, ast.Name):
        traced = normalize_source(trace_fn(node))
        if isinstance(traced, CallResult):
            result_source = normalize_source(traced.result_source)
            if (isinstance(result_source, DerivedResult)
                    and result_source.kind == "iterator"
                    and result_source.sources):
                return result_source.sources[0]
        return ContainerIter(node.id)

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        sources = [_builtin_value_source(elt, trace_fn) for elt in node.elts]
        if sources:
            return make_source_set(sources, origin="builtin_element")
        return UnknownSource("empty iterable")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("iter", "reversed") and node.args:
            return _iterable_element_source(node.args[0], trace_fn)
        if node.func.id == "enumerate" and node.args:
            # enumerate yields Python tuples regardless of the item owner.
            return "python"

    traced = normalize_source(trace_fn(node))
    if isinstance(traced, CallResult):
        result_source = normalize_source(traced.result_source)
        if (isinstance(result_source, DerivedResult)
                and result_source.kind == "iterator"
                and result_source.sources):
            return result_source.sources[0]
    if traced is not None:
        return ContainerIter(traced)
    return UnknownSource("iterable element")


## Resolve element source for min/max/next arguments.
#
#  min/max with multiple values select from those values.  Their one-argument
#  forms and next() select an iterable element.  A default value is a separate
#  possible result and therefore participates in the same SourceSet.
#  @param name Bare builtin name.
#  @param call_node The ast.Call node.
#  @param trace_fn Callable to trace an AST expression.
#  @return DerivedResult("element", ...) or UnknownSource.
def _element_source(name, call_node, trace_fn):
    sources = []
    if name in ("min", "max") and len(call_node.args) > 1:
        sources.extend(_builtin_value_source(arg, trace_fn)
                       for arg in call_node.args)
    elif call_node.args:
        sources.append(_iterable_element_source(call_node.args[0], trace_fn))

    if name in ("min", "max"):
        for keyword in call_node.keywords:
            if keyword.arg == "default":
                sources.append(_builtin_value_source(keyword.value, trace_fn))
    elif name == "next" and len(call_node.args) > 1:
        sources.append(_builtin_value_source(call_node.args[1], trace_fn))

    if not sources:
        return UnknownSource("element")
    return DerivedResult(
        "element", (make_source_set(sources, origin="builtin_element"),))


## Return the result_source for a known builtin callable.
#
#  @param name Bare builtin name.
#  @param call_node The ast.Call node (for argument tracing).
#  @param trace_fn Callable to trace an AST expression to its source.
#  @return "python", UnknownSource, DerivedResult, module name string, or None.
def _resolve_builtin_result(name, call_node, trace_fn):
    if not isinstance(name, str):
        return None
    if name in _BUILTIN_PYTHON_OWNED_RESULT:
        return "python"
    if name in _BUILTIN_ARBITRARY_RESULT:
        return UnknownSource(name)
    if name in _BUILTIN_ELEMENT_DERIVED and call_node and call_node.args:
        return _element_source(name, call_node, trace_fn)
    if name == "type" and call_node and call_node.args:
        arg_node = call_node.args[0]
        arg_source = trace_fn(arg_node)
        if isinstance(arg_node, ast.Call):
            # type(A()) — resolve the constructor's callee identity
            from .sources import normalize_source as _ns
            callee_id = None
            if isinstance(arg_node.func, ast.Name):
                callee_id = arg_node.func.id
            elif isinstance(arg_node.func, ast.Attribute):
                callee_id = trace_fn(arg_node)
            if callee_id is not None:
                callee_norm = _ns(callee_id)
                if isinstance(callee_norm, CallResult) and isinstance(callee_norm.callee, str):
                    callee_id = callee_norm.callee
                return DerivedResult("type_of", (callee_id,))
        if arg_source is not None:
            return DerivedResult("type_of", (arg_source,))
        return UnknownSource("type")
    if name == "__import__" and call_node and call_node.args:
        first_arg = call_node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value.split(".")[0]
        return UnknownSource("__import__")
    if name in ("iter", "reversed") and call_node and call_node.args:
        element = _iterable_element_source(call_node.args[0], trace_fn)
        return DerivedResult("iterator", (element,))
    if name in _BUILTIN_PROTOCOL_DERIVED:
        return UnknownSource(name)
    return None


## Check if a Call node is a defaultdict(list) call with a statically
#  known default factory (list/dict/set/tuple/str).
def _is_defaultdict_itemkind(node):
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name):
        return False
    if node.func.id != "defaultdict" or len(node.args) < 1:
        return False
    factory = node.args[0]
    return isinstance(factory, ast.Name) and factory.id in ("list", "dict", "set", "tuple", "str")


## Return the concrete Python container kind produced by an expression.
#  @param node Assignment right-hand side AST node.
#  @return Container kind string or None.
def _container_kind(node):
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "str"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("list", "dict", "set", "tuple", "str"):
            return node.func.id
        if node.func.id == "defaultdict":
            return "dict"
    return None


## Return the concrete kind produced by subscripting a container expression.
#  Dict literals qualify only when every value has the same known kind.
#  @param node Assignment right-hand side AST node.
#  @return Item kind string or None.
def _container_item_kind(node):
    if isinstance(node, ast.Dict) and node.values:
        kinds = [_container_kind(value) for value in node.values]
        if kinds[0] is not None and all(kind == kinds[0] for kind in kinds):
            return kinds[0]
    if _is_defaultdict_itemkind(node):
        return node.args[0].id
    return None
from .types import FileAnalysis, ApiCall

# 1.0.5 P1: known conversion targets.  Method calls (to_numpy())
# change the result type; bare attribute reads (values) also
# change the result type.  Bare method references (df.to_numpy
# without calling) are NOT conversions.
_CONVERSION_METHOD_TARGETS = {
    ("pandas", "to_numpy"): "numpy",
}
_CONVERSION_ATTRIBUTE_TARGETS = {
    ("pandas", "values"): "numpy",
}

# Verified attribute-result contracts.  The receiver owner must already be
# import-backed; matching an attribute name alone is never sufficient.
_ATTRIBUTE_RESULT_OWNER_CONTRACTS = {
    ("bs4", "text"): (
        "python", "public-api:bs4.PageElement.text", PythonShape("str")),
    ("requests", "text"): (
        "python", "public-api:requests.Response.text", PythonShape("str")),
    ("spacy", "text"): (
        "python", "public-api:spacy Token/Span/Doc.text",
        PythonShape("str")),
    ("xml", "text"): (
        "python", "python-stdlib:xml.etree.ElementTree.Element.text",
        PythonShape("str")),
}

# Verified result-object contracts for import-backed calls.  The callable keeps
# its own library owner; each contract applies only to the object returned
# across an assignment or chained-call boundary.  Values contain
# (result_owner, evidence).  Evidence points to a checked Python/stdlib
# contract, a public API contract, or a committed runtime probe.
_RESULT_OWNER_CONTRACTS = {
    ("Box2D", "CreateDynamicBody"): (
        "Box2D", "probe:parameter_receiver_ownership"),
    ("Box2D", "CreateStaticBody"): (
        "Box2D", "probe:parameter_receiver_ownership"),
    ("scipy", "cdist"): (
        "numpy", "probe:receiver_ownership"),
    # svd() returns a Python tuple whose unpacked items are NumPy arrays.
    ("scipy", "svd"): (
        "python", "probe:machine_learning_svd"),
    ("scipy", "bisplev"): (
        "numpy", "public-api:scipy.interpolate.bisplev"),
    ("numpy", "dot"): (
        "numpy", "probe:receiver_ownership"),
    ("numpy", "reshape"): (
        "numpy", "public-api:numpy.reshape"),
    ("seaborn", "barplot"): (
        "matplotlib", "public-api:seaborn.barplot"),
    ("seaborn", "stripplot"): (
        "matplotlib", "public-api:seaborn.stripplot"),
    ("seaborn", "swarmplot"): (
        "matplotlib", "public-api:seaborn.swarmplot"),
    ("matplotlib", "figure"): (
        "matplotlib", "public-api:matplotlib.pyplot.figure"),
    ("matplotlib", "gca"): (
        "matplotlib", "public-api:matplotlib.pyplot.gca"),
    ("matplotlib", "gcf"): (
        "matplotlib", "public-api:matplotlib.pyplot.gcf"),
    ("matplotlib", "subplot"): (
        "matplotlib", "public-api:matplotlib.pyplot.subplot"),
    # subplots() returns a Python tuple.  Its second unpacked item may be a
    # Matplotlib Axes or a NumPy array, so no uniform item owner is claimed.
    ("matplotlib", "subplots"): (
        "python", "public-api:matplotlib.pyplot.subplots"),
    ("matplotlib", "add_subplot"): (
        "matplotlib", "public-api:matplotlib.figure.Figure.add_subplot"),
    ("skimage", "downscale_local_mean"): (
        "numpy", "probe:ground_truth/probes/round6_probe.py"),
    ("torchvision", "to_tensor"): (
        "torch", "public-api:torchvision.transforms.functional.to_tensor"),
    # Stable standard-library contracts.  Both functions return a
    # Python-provided str/bytes object, not an object owned by the module.
    ("json", "dumps"): (
        "python", "python-stdlib:json.dumps", PythonShape("str")),
    ("json", "load"): (
        "python", "python-stdlib:json.load"),
    ("json", "loads"): (
        "python", "python-stdlib:json.loads"),
    ("re", "sub"): (
        "python", "python-stdlib:re.sub", PythonShape("str")),
    ("re", "split"): (
        "python", "python-stdlib:re.split", PythonShape("list", "str")),
    ("re", "group"): (
        "python", "python-stdlib:re.Match.group", PythonShape("str")),
    ("re", "compile"): (
        "re", "python-stdlib:re.compile"),
    ("re", "match"): (
        "re", "python-stdlib:re.match"),
    ("re", "search"): (
        "re", "python-stdlib:re.search"),
    ("re", "fullmatch"): (
        "re", "python-stdlib:re.fullmatch"),
}
_VERIFIED_RESULT_OWNERS = frozenset(
    contract[0] for contract in _RESULT_OWNER_CONTRACTS.values()
)

# Owners of elements yielded by selected import-backed iterator calls. This is
# deliberately separate from _RESULT_OWNER_CONTRACTS: the iterator object and
# each yielded object do not necessarily have the same ownership semantics.
_ITERATOR_ELEMENT_OWNER_MAP = {
    ("re", "finditer"): "re",
    ("glob", "glob"): ("python", PythonShape("str")),
    ("glob", "iglob"): ("python", PythonShape("str")),
    ("os", "listdir"): ("python", PythonShape("str")),
}

# Owners of items selected from selected call results.  Keep this separate
# from _RESULT_OWNER_CONTRACTS because the aggregate result may be a Python
# tuple while its destructured or indexed items are import-backed objects.
_RESULT_ITEM_OWNER_CONTRACTS = {
    ("scipy", "svd"): "numpy",
    ("GPy", "predict"): "numpy",
    ("re", "split"): "python",
}

# Verified predicates that narrow a receiver owner in their true branch.
# The evidence is part of the contract so these rules remain distinguishable
# from method-name guessing.
_TYPE_GUARD_OWNER_CONTRACTS = {
    ("scipy", "issparse"): (
        "scipy", "public-api:scipy.sparse.issparse"),
}

# Verified callback-parameter contracts.  Each key is
# (library, callable, callback argument index, callback parameter index).
_CALLBACK_PARAMETER_OWNER_CONTRACTS = {
    ("re", "sub", 1, 0): (
        "re", "python-stdlib:re.sub replacement callback"),
    ("re", "subn", 1, 0): (
        "re", "python-stdlib:re.subn replacement callback"),
}

# Item kind produced by indexing selected builtin-method results.  Keep this
# table limited to contracts guaranteed by Python itself; arbitrary local
# methods with the same name do not enter this path unless their receiver kind
# is independently known.
_BUILTIN_METHOD_RESULT_ITEM_KINDS = {
    ("str", "split"): "str",
    ("str", "rsplit"): "str",
}

def _match_result_owner(top, func_name):
    """Return the verified owner of an import-backed call's result object."""
    if top is None:
        return None
    for (lib_prefix, fn), contract in _RESULT_OWNER_CONTRACTS.items():
        if (fn == func_name
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return contract[0]
    return None


def _has_result_owner_contract(func_name):
    """Return whether any verified result contract covers a method name."""
    return any(
        contract_name == func_name
        for _, contract_name in _RESULT_OWNER_CONTRACTS
    )


def _match_attribute_result_owner(top, attribute):
    """Return the verified owner of an import-backed attribute's value."""
    if top is None:
        return None
    for (lib_prefix, name), contract in (
            _ATTRIBUTE_RESULT_OWNER_CONTRACTS.items()):
        if (name == attribute
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return contract[0]
    return None


def _match_attribute_python_shape(top, attribute):
    """Return the concrete Python shape from a verified attribute contract."""
    if top is None:
        return None
    for (lib_prefix, name), contract in (
            _ATTRIBUTE_RESULT_OWNER_CONTRACTS.items()):
        if (name == attribute
                and len(contract) >= 3
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return contract[2]
    return None


def _match_result_python_shape(top, func_name):
    """Return the concrete Python shape from a verified result contract."""
    if top is None:
        return None
    for (lib_prefix, fn), contract in _RESULT_OWNER_CONTRACTS.items():
        if (fn == func_name
                and len(contract) >= 3
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return contract[2]
    return None


def _is_verified_result_owner(owner):
    """Return whether owner is produced by a verified result contract."""
    return owner in _VERIFIED_RESULT_OWNERS


def _match_result_item_owner(top, func_name):
    """Return a uniform owner for destructured or indexed call-result items."""
    if top is None:
        return None
    for (lib_prefix, fn), owner in _RESULT_ITEM_OWNER_CONTRACTS.items():
        if (fn == func_name
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return owner
    return None


def _match_iterator_element_owner(top, func_name):
    """Return the owner of elements from a known import-backed iterator."""
    if top is None:
        return None
    for (lib_prefix, fn), contract in _ITERATOR_ELEMENT_OWNER_MAP.items():
        if (fn == func_name
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return contract[0] if isinstance(contract, tuple) else contract
    return None


def _match_iterator_element_shape(top, func_name):
    """Return the Python shape of elements from a verified iterator."""
    if top is None:
        return None
    for (lib_prefix, fn), contract in _ITERATOR_ELEMENT_OWNER_MAP.items():
        if (fn == func_name
                and isinstance(contract, tuple)
                and (top == lib_prefix
                     or top.startswith(lib_prefix + "."))):
            return contract[1]
    return None


## Check whether every possible return source is Python-owned.
#
#  SourceSet values represent branch-dependent returns.  A local call may use
#  the Python result contract only when every branch independently proves the
#  same owner; mixed or unresolved branches remain conservative.
#  @param source Return source or SourceSet.
#  @return True when all possible sources are exactly "python".
def _is_uniform_python_result(source):
    source = normalize_source(source)
    if source == "python" or isinstance(source, PythonShape):
        return True
    if isinstance(source, SourceSet) and source.sources:
        return all(_is_uniform_python_result(item)
                   for item in source.sources)
    return False


## Check whether any possible return source is Python-owned.
#  @param source Return source or SourceSet.
#  @return True when at least one branch is exactly "python".
def _has_python_result(source):
    source = normalize_source(source)
    if source == "python" or isinstance(source, PythonShape):
        return True
    if isinstance(source, SourceSet):
        return any(_has_python_result(item) for item in source.sources)
    return False


## Return one concrete Python shape when every return branch agrees.
#
#  @param source Return source or SourceSet.
#  @return PythonShape when all branches have the same shape, otherwise None.
def _uniform_python_shape(source):
    source = normalize_source(source)
    if isinstance(source, PythonShape):
        return source
    if isinstance(source, CallResult):
        result_source = normalize_source(source.result_source)
        return result_source if isinstance(result_source, PythonShape) else None
    if isinstance(source, SourceSet) and source.sources:
        shapes = [_uniform_python_shape(item) for item in source.sources]
        if (all(shape is not None for shape in shapes)
                and all(shape == shapes[0] for shape in shapes[1:])):
            return shapes[0]
    return None

# 1.0.5 P1: numpy ufuncs that preserve the receiver's type when
# applied to pandas objects.  Probe-backed: np.log(pd.Series)
# returns pd.Series.
_RECEIVER_PRESERVE_UFUNCS = frozenset({
    "log", "exp", "sqrt", "abs", "divide",
})

# Methods known to be valid on compare-result objects, keyed by
# the result owner.  Only these (owner, method) pairs allow a
# compare-receiver call to be classified as that owner.
_COMPARE_RESULT_METHODS = {
    "numpy": frozenset(["any", "all"]),
}


## AST visitor that traces all symbols and API calls in a single Python file.
#
#
#  Walks the AST to:
#  - Record import mappings and their aliases
#  - Track assignments, function/class definitions, decorators
#  - Resolve with/for return-value flows
#  - Handle container indexing, class inheritance, method resolution
#  - Detect and classify all API call expressions
class SingleFileAnalyzer(ast.NodeVisitor):
    ## Initialize the analyzer with empty state.
    #  @param module_name Optional dotted module name for resolving relative imports.
    #  @param is_package Whether the file is a package __init__.py.
    def __init__(self, module_name=None, is_package=False, file_path=""):
        self.module_name = module_name
        self.is_package = is_package
        self._file_path = file_path
        self.return_sources = {}
        # Literal return containers own their aggregate, not their elements.
        self.return_element_sources = {}
        # Positional tuple return summaries are consumed only by the project
        # call graph. Keep them out of return_sources, whose legacy resolver
        # expects one whole-result source.
        self.call_graph_return_sources = {}
        self.call_graph_return_values = {}
        self.call_graph_yield_sources = {}
        self.symbols = SymbolTable(self.return_sources)
        self.api_calls = []
        self.comprehension_targets = set()
        self.attr_accesses = []
        self.local = set()
        self._func_stack = []
        self._class_stack = []
        self._super_base_path_stack = []
        self._seen_api_call_ids = set()
        self._receiver_owner_guards = []
        self.defined_functions = set()
        self.function_params = {}
        self.parameter_sources = {}
        self._assigned_call_sources = {}
        self.container_items = {}
        self.homogeneous_container_items = {}
        self.homogeneous_container_value_sources = {}
        # Positional sources for homogeneous tuple/list-comprehension items.
        # Keys are module-level container names; values are source tuples.
        self.homogeneous_container_tuple_items = {}
        self.container_lengths = {}
        self.container_kinds = {}  # 1.0.5 P1: name -> "list"|"dict"|"set"|"tuple"|"str"
        self.container_item_kinds = {}  # 1.0.5 P1+: name -> "list"|"dict"|... for defaultdict(list) etc.
        self.container_set_sources = {}
        self.class_methods = {}
        self.class_bases = {}
        self.instance_attrs = {}
        self.instance_attr_kinds = {}
        self.instance_attr_item_kinds = {}
        self.instance_attr_item_fields = {}
        self._constructor_kwargs_contracts = {}
        self._local_instance_field_sources = {}
        self._local_instance_field_shapes = {}
        self._local_instance_field_shadowed = {}
        self.class_attr_kinds = {}
        self._class_receiver_stack = []
        self._container_item_kind_conflicts = set()
        self.import_from_symbols = {}
        self.wildcard_modules = []
        self.import_aliases = set()
        self._import_binding_sources = {}
        self.call_sites = {}
        self.call_assign_funcs = {}
        self._assignment_counter = 0
        self._global_names = set()
        self.call_site_objects = []
        self.symbol_refs = []
        self.module_scope = Scope(SCOPE_MODULE, self.module_name or "<module>")
        self.scope_stack = [self.module_scope]
        ## Call-graph facts (Phase 7B-full PR1: read-only collection).
        self.module_cg = ModuleCallGraph(module=module_name or "")
        ## Stack of FunctionId for tracking the current caller context.
        self._caller_stack = [FunctionId(module_name or "", "<module>")]
        ## Map from RHS top-level expression node id -> list of target names.
        ## Only the outermost RHS call (not nested inner calls) consumes targets.
        self._literal_values = {}
        self._pending_call_targets_by_node = {}
        self._argparse_parsers = {}
        self._argparse_destination_shapes = {}
        self._iterable_binding_sources = {}
        self._iterated_append_sources = {}
        self._iterated_append_tuple_sources = {}
        self._iterated_append_tuple_conflicts = set()
        self._attribute_append_tuple_sources = {}
        self._attribute_append_tuple_conflicts = set()
        self._subscripted_import_attribute_receivers = set()
        self.external_method_overrides = {}
        self._mapping_facts = MappingFacts(
            self._mapping_binding,
            lambda node: FunctionId(
                self.module_name or "", self._local_lambda_key(node)),
            lambda name: self.current_scope().lookup(
                name, skip_parent_classes=True),
            self._expression_python_shape)
        self._mapping_edges = []
        self._mapping_local_names = {}
        self._mapping_blocked_names = set()
        self._finite_name_guards = []
        self._finite_namespace_stability = {}
        self._module_tree = None

    ## Visit with mapping joins disabled for loop/exception-assigned names.
    #  Existing owner and protocol analysis remains unchanged.
    #  @param node AST node.
    def visit(self, node):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try)):
            return super().visit(node)
        names = mapping_bound_names([node])
        scope = self.current_scope()
        previous = self._mapping_blocked_names
        self._mapping_blocked_names = previous | {
            (id(scope), name) for name in names}
        for name in names:
            binding = scope.lookup(name, skip_parent_classes=True)
            self._mapping_facts.invalidate(
                getattr(binding, "mapping_value", None), rebind=True)
        try:
            return super().visit(node)
        finally:
            self._mapping_blocked_names = previous

    ## Complete mapping call evidence after all possible mutations are seen.
    #  @param node Module AST node.
    def visit_Module(self, node):
        self._module_tree = node
        self.generic_visit(node)
        for edge, value in self._mapping_edges:
            edge.mapping_targets, edge.mapping_targets_complete = value.targets()

    ## Look up mapping facts without importing a deferred closure's bindings.
    #  @param name Lexical name.
    #  @return Current or module Binding, otherwise None.
    def _mapping_binding(self, name):
        scope = self.current_scope()
        if (id(scope), name) in self._mapping_blocked_names:
            return None
        if (name in self._mapping_local_names.get(id(scope), set())
                and name not in scope.bindings):
            return None
        binding = self.current_scope().lookup(name, skip_parent_classes=True)
        if (binding is not None
                and binding.scope_kind != SCOPE_MODULE
                and self.current_scope().bindings.get(name) is not binding):
            return None
        return binding

    ## Store mapping facts separately from the assigned object's owner.
    #  @param target Assignment target.
    #  @param value Captured mapping value.
    def _bind_mapping_value(self, target, value):
        if isinstance(target, ast.Name):
            binding = self.current_scope().bindings.get(target.id)
            if binding is not None:
                blocked = ((id(self.current_scope()), target.id)
                           in self._mapping_blocked_names)
                binding.mapping_value = None if blocked else value
        elif isinstance(target, ast.Subscript):
            self._mapping_facts.write(target)
            self._mapping_facts.invalidate(value)
        elif isinstance(target, ast.Attribute):
            self._mapping_facts.escape(target.value)
            self._mapping_facts.invalidate(value)

    ## Return the current innermost scope.
    def current_scope(self):
        return self.scope_stack[-1]

    ## Push a new scope onto the stack.
    #  @param kind Scope kind constant.
    #  @param name Human-readable scope name.
    #  @return The new Scope.
    def push_scope(self, kind, name):
        parent = self.current_scope()
        scope = Scope(kind, name, parent)
        self.scope_stack.append(scope)
        return scope

    ## Pop the current scope from the stack.
    #  @return The popped Scope.
    def pop_scope(self):
        return self.scope_stack.pop()

    ## Return the stable analyzer-local identity of a lexical binding.
    #  @param binding Binding created by this analyzer.
    #  @return Monotonic assignment index unique within the analyzed file.
    def _binding_key(self, binding):
        return binding.assignment_index

    ## Bind a name in the current lexical scope and compatibility symbol table.
    #
    #  Only module-scope and explicit global bindings enter self.symbols.
    #  @param name Symbol name.
    #  @param source Source value.
    #  @param node Optional AST node for position info.
    #  @param kind Optional symbol kind for provenance ("variable", "parameter", "attribute").
    def _bind_target_name(self, name, source, node=None, kind="variable",
                          container_kind="", container_item_kind="",
                          callable_key="", container_item_fields=None):
        if self.current_scope().kind == SCOPE_MODULE or name in self._global_names:
            old = self.current_scope().lookup(name, skip_parent_classes=True)
            self._mapping_facts.invalidate(
                getattr(old, "mapping_value", None), rebind=True)
        self._assignment_counter += 1
        lineno = getattr(node, "lineno", 0) if node is not None else 0
        col = getattr(node, "col_offset", 0) if node is not None else 0
        self.current_scope().bind(
            name, source, lineno, col, self._assignment_counter,
            container_kind=container_kind,
            container_item_kind=container_item_kind,
            callable_key=callable_key,
            binding_kind=kind,
            container_item_fields=container_item_fields)
        if (name in self._global_names
                or self.current_scope().kind == SCOPE_MODULE):
            self.symbols.add(name, source)
        if name.startswith("self.") and self._class_stack:
            attr_key = (self._class_stack[-1], name)
            self.instance_attrs[attr_key] = source
            if container_kind:
                self.instance_attr_kinds[attr_key] = container_kind
            else:
                self.instance_attr_kinds.pop(attr_key, None)
            if container_item_kind:
                self.instance_attr_item_kinds[attr_key] = container_item_kind
            else:
                self.instance_attr_item_kinds.pop(attr_key, None)
            if container_item_fields:
                self.instance_attr_item_fields[attr_key] = dict(
                    container_item_fields)
            else:
                self.instance_attr_item_fields.pop(attr_key, None)
        if (self.current_scope().kind == SCOPE_CLASS
                and self._class_stack and "." not in name):
            attr_key = (self._class_stack[-1], name)
            if container_kind:
                self.class_attr_kinds[attr_key] = container_kind
            else:
                self.class_attr_kinds.pop(attr_key, None)
        if kind:
            self._add_symbol_ref(name, source, kind, node)

    ## Normalize a class-qualified instance field target.
    #
    #  A local singleton may initialize an instance field through a class
    #  holder, for example ``MySQL.__instance.connection = pymysql.connect``.
    #  The later method body still observes that field as ``self.connection``.
    #  Preserve only this syntactic binding relationship; do not infer an
    #  external return type from the field name or method name.
    #  @param name Dotted assignment target.
    #  @return ``self.<field>`` or None when the target is unrelated.
    def _instance_attribute_target_name(self, name):
        if not self._class_stack or not isinstance(name, str):
            return None
        class_name = self._class_stack[-1]
        prefixes = (class_name + ".", "cls.")
        prefix = next(
            (candidate for candidate in prefixes
             if name.startswith(candidate)), None)
        if prefix is None:
            return None
        if prefix == "cls.":
            binding = self.current_scope().lookup(
                "cls", skip_parent_classes=True)
            if (binding is None
                    or binding.binding_kind != "parameter"):
                return None
        parts = name[len(prefix):].split(".")
        if parts and parts[0] == "__instance":
            parts = parts[1:]
        if not parts:
            return None
        return "self." + ".".join(parts)

    ## Look up a name in the lexical scope chain or return the name as-is.
    #
    #  Unified helper so that trace_source, get_base, and _resolve_call_receiver
    #  all use the same scope-aware resolution.
    #  @param name The raw AST name string.
    #  @return Scope binding source, or the name itself when not found.
    def _lookup_name_source(self, name):
        binding = self.current_scope().lookup(name, skip_parent_classes=True)
        if binding is not None:
            if binding.source == "local":
                return binding.source
            if (binding.binding_kind == "import"
                    and binding.scope_kind != SCOPE_MODULE):
                return binding.source
            if (name in self.import_aliases
                    and isinstance(binding.source, str)
                    and '.' not in binding.source):
                return name
            return binding.source
        return name

    ## Return assignment metadata for a name in the active lexical scope.
    #  A found binding with no metadata is authoritative and prevents a
    #  same-name binding from another scope leaking through legacy maps.
    #  @param name Receiver variable name.
    #  @param item Whether to request the subscript item kind.
    #  @return Container kind string or None.
    def _lookup_container_kind(self, name, item=False):
        binding = self.current_scope().lookup(
            name, skip_parent_classes=True)
        if binding is not None:
            attr = "container_item_kind" if item else "container_kind"
            return getattr(binding, attr, "") or None
        # Compatibility maps are file-wide and can contain a same-name local
        # from a previously visited function. Only module code may use their
        # fallback; nested scopes require a lexical Binding as evidence.
        if self.current_scope().kind != SCOPE_MODULE:
            return None
        if item:
            return self.container_item_kinds.get(name)
        return self.container_kinds.get(name)

    ## Return field shapes for elements of a statically known container.
    #  @param node Container expression or bound name.
    #  @return Mapping of literal field names to shape tuples.
    def _expression_container_item_fields(self, node):
        if isinstance(node, (ast.List, ast.Tuple)):
            field_sets = []
            for element in node.elts:
                fields = self._literal_dict_field_shapes(element)
                if not fields:
                    return {}
                field_sets.append(fields)
            if field_sets and all(fields == field_sets[0]
                                  for fields in field_sets):
                return dict(field_sets[0])
            return {}
        if isinstance(node, ast.Name):
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            return dict(getattr(binding, "container_item_fields", {}) or {})
        if isinstance(node, ast.Attribute):
            name = self._attribute_name(node)
            if name and name.startswith("self.") and self._class_stack:
                return dict(self.instance_attr_item_fields.get(
                    (self._class_stack[-1], name), {}) or {})
        return {}

    ## Return positional sources for a homogeneous tuple/list comprehension.
    #
    #  The fact is intentionally limited to a comprehension whose element is
    #  a tuple or list and whose every field has an explicit source. It lets
    #  ``for value, label in pairs`` preserve the source of ``value`` without
    #  inferring ownership from the loop variable or method name.
    #  @param node Candidate list comprehension.
    #  @return Tuple of field sources, or None when incomplete.
    def _expression_tuple_item_sources(self, node):
        if not isinstance(node, ast.ListComp):
            return None
        element = node.elt
        if not isinstance(element, (ast.Tuple, ast.List)):
            return None
        sources = []
        for field in element.elts:
            source = self.trace_source(field)
            if source is None:
                source = self.get_base(field)
            if source is None:
                return None
            sources.append(normalize_source(source))
        return tuple(sources)

    ## Return module-level tuple-field facts for an active container binding.
    #  @param name Container name.
    #  @return Tuple of sources, or None when unavailable or shadowed.
    def _lookup_tuple_item_sources(self, name):
        binding = self.current_scope().lookup(
            name, skip_parent_classes=True)
        if binding is not None and binding.scope_kind != SCOPE_MODULE:
            return None
        return self.homogeneous_container_tuple_items.get(name)

    ## Return the lexical identity of an attribute-backed container.
    #
    #  The root binding's monotonic assignment index keeps same-spelled local
    #  objects in separate functions isolated without relying on reusable
    #  Python object ids. Attribute facts are shared across functions only
    #  when their root resolves to the same enclosing binding.
    #  @param node Attribute expression naming the container.
    #  @return Tuple of root binding identity and attribute path, or None.
    def _attribute_container_key(self, node):
        chain = self._attribute_chain_list(node)
        if not chain or len(chain) < 2:
            return None
        binding = self.current_scope().lookup(
            chain[0], skip_parent_classes=True)
        if binding is None:
            return None
        return (binding.assignment_index, tuple(chain[1:]))

    ## Return tuple-field facts recorded for an attribute-backed list.
    #  @param node Attribute expression naming the list.
    #  @return TupleSource fields, or None when unresolved or conflicting.
    def _lookup_attribute_tuple_item_sources(self, node):
        key = self._attribute_container_key(node)
        if key is None or key in self._attribute_append_tuple_conflicts:
            return None
        source = self._attribute_append_tuple_sources.get(key)
        return source.items if isinstance(source, TupleSource) else None

    ## Invalidate tuple-field facts when an attribute container is rebound.
    #  @param node Attribute assignment target naming the container.
    def _invalidate_attribute_tuple_item_sources(self, node):
        key = self._attribute_container_key(node)
        if key is None:
            return
        self._attribute_append_tuple_sources.pop(key, None)
        self._attribute_append_tuple_conflicts.discard(key)

    ## Check whether two tuple item facts have the same positional owners.
    #  Call locations and expression spellings are not ownership differences.
    #  Unresolved positions never converge.
    #  @param left Existing TupleSource fact.
    #  @param right Newly observed TupleSource fact.
    #  @return True when every field has one matching resolved owner.
    def _tuple_item_owners_match(self, left, right):
        if (not isinstance(left, TupleSource)
                or not isinstance(right, TupleSource)
                or len(left.items) != len(right.items)):
            return False
        for left_item, right_item in zip(left.items, right.items):
            left_owner = self._structured_source_owner_top(left_item)
            right_owner = self._structured_source_owner_top(right_item)
            if (left_owner in (None, "unknown", "")
                    or left_owner != right_owner):
                return False
        return True

    ## Infer field shapes from a literal dictionary element.
    #  @param node Candidate dictionary AST node.
    #  @return Mapping of literal string keys to shape tuples.
    def _literal_dict_field_shapes(self, node):
        if not isinstance(node, ast.Dict):
            return {}
        fields = {}
        for key_node, value_node in zip(node.keys, node.values):
            if (not isinstance(key_node, ast.Constant)
                    or not isinstance(key_node.value, str)):
                continue
            kind, item_kind = self._expression_container_shape(value_node)
            if not kind:
                continue
            fields[key_node.value] = (kind, item_kind)
        return fields

    ## Return the shape of a literal-key field on a known element.
    #  @param node Subscript AST node.
    #  @return (container kind, item kind), or empty strings.
    def _expression_subscript_field_shape(self, node):
        if not isinstance(node, ast.Subscript):
            return ("", "")
        key = self._get_slice(node.slice)
        if not isinstance(key, str):
            return ("", "")
        fields = self._expression_container_item_fields(node.value)
        shape = fields.get(key)
        if shape is None:
            return ("", "")
        return shape

    ## Resolve a structured source to one uniform owner within the file.
    #
    #  This follows existing provenance only. It does not infer ownership from
    #  an attribute or method name, and mixed SourceSet owners stay unresolved.
    #  @param source Source value to inspect.
    #  @param seen Recursion guard for cyclic structured sources.
    #  @return Uniform owner string or None.
    def _structured_source_owner_top(self, source, seen=None):
        source = normalize_source(source)
        visited = set(seen or set())
        key = (type(source).__name__, source_display(source))
        if key in visited:
            return None
        visited.add(key)

        if isinstance(source, str):
            if source in ("local", "python", "unknown", ""):
                return source or None
            return self.symbols.get_top(source) or source
        if isinstance(source, PythonShape):
            return "python"
        if isinstance(source, UnknownSource):
            return "unknown"
        if isinstance(source, CallResult):
            if source.result_source is not None:
                return self._structured_source_owner_top(
                    source.result_source, visited)
            return self._structured_source_owner_top(
                source.callee, visited)
        if isinstance(source, InstanceMethod):
            return self._structured_source_owner_top(
                source.receiver, visited)
        if isinstance(source, ContainerIter):
            return self._structured_source_owner_top(
                source.container, visited)
        if isinstance(source, ContainerItem):
            return self._structured_source_owner_top(
                source.container, visited)
        if isinstance(source, SourceSet):
            owners = {
                self._structured_source_owner_top(item, set(visited))
                for item in source.sources
            }
            owners.discard(None)
            if len(owners) == 1:
                return next(iter(owners))
        return None

    ## Infer a Python-provided container shape from local expression evidence.
    #
    #  This follows only language-level facts: literals, lexical bindings,
    #  slicing, homogeneous items, and builtin methods on an independently
    #  known receiver kind. It never infers a receiver kind from a method name.
    #  @param node Value expression.
    #  @return Tuple of (container_kind, container_item_kind), or ("", "").
    def _expression_container_shape(self, node):
        direct_kind = _container_kind(node)
        if direct_kind is not None:
            return (direct_kind, _container_item_kind(node) or "")
        if (isinstance(node, ast.Constant)
                and type(node.value) in (bool, int, float, complex, bytes)):
            return (type(node.value).__name__, "")

        if isinstance(node, ast.JoinedStr):
            return ("str", "")

        if isinstance(node, ast.Name):
            return (
                self._lookup_container_kind(node.id) or "",
                self._lookup_container_kind(node.id, item=True) or "",
            )

        if isinstance(node, ast.Attribute):
            name = self._attribute_name(node)
            if name and name.startswith("self.") and self._class_stack:
                key = (self._class_stack[-1], name)
                return (
                    self.instance_attr_kinds.get(key, ""),
                    self.instance_attr_item_kinds.get(key, ""),
                )
            if name:
                binding = self.current_scope().lookup(
                    name, skip_parent_classes=True)
                if binding is not None:
                    return (
                        getattr(binding, "container_kind", "") or "",
                        getattr(binding, "container_item_kind", "") or "",
                    )
                parts = name.split(".", 1)
                if len(parts) == 2:
                    for scope_key in self._local_instance_field_scope_keys(
                            parts[0]):
                        field_shape = self._local_instance_field_shapes.get(
                            (scope_key, parts[0], parts[1]))
                        if field_shape is not None:
                            return field_shape
            receiver_top = self._expr_receiver_top(node.value)
            if receiver_top is None:
                receiver_top = self._structured_source_owner_top(
                    self.trace_source(node.value))
            shape = _match_attribute_python_shape(
                receiver_top, node.attr)
            if shape is not None:
                return (shape.kind, shape.item_kind)
            return ("", "")

        if isinstance(node, ast.Subscript):
            field_kind = self._expression_subscript_field_shape(node)
            if field_kind[0]:
                return field_kind
            value_kind, item_kind = self._expression_container_shape(
                node.value)
            if isinstance(node.slice, ast.Slice):
                if value_kind in ("list", "tuple", "str"):
                    return (value_kind, item_kind)
                return ("", "")
            if item_kind:
                return (item_kind, "")
            if value_kind == "str":
                return ("str", "")
            return ("", "")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            left = self._expression_container_shape(node.left)
            if left[0] == "str":
                return ("str", "")
            return ("", "")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._expression_container_shape(node.left)
            right = self._expression_container_shape(node.right)
            if (left[0] in ("list", "tuple", "str")
                    and left == right):
                return left
            return ("", "")

        if isinstance(node, ast.IfExp):
            body = self._expression_container_shape(node.body)
            orelse = self._expression_container_shape(node.orelse)
            return body if body[0] and body == orelse else ("", "")

        if isinstance(node, ast.Call):
            call_key = self.get_base(node, call_lookup=True)
            if isinstance(call_key, str):
                local_shape = _uniform_python_shape(
                    self.return_sources.get(call_key))
                if local_shape is not None:
                    return (local_shape.kind, local_shape.item_kind)
            func_top, func_name = self._resolve_func_top(node.func)
            if (func_top is None
                    and isinstance(node.func, ast.Attribute)):
                func_top = self._expr_receiver_top(node.func.value)
                func_name = node.func.attr
            if (func_top is None
                    and isinstance(node.func, ast.Attribute)):
                method_source = normalize_source(
                    self._resolve_methods(node))
                if isinstance(method_source, InstanceMethod):
                    receiver = normalize_source(method_source.receiver)
                    if isinstance(receiver, str):
                        func_top = (
                            self.symbols.get_top(receiver) or receiver)
                    elif isinstance(receiver, CallResult):
                        result_source = normalize_source(
                            receiver.result_source)
                        if isinstance(result_source, str):
                            func_top = result_source
                        elif isinstance(receiver.callee, str):
                            func_top = (
                                self.symbols.get_top(receiver.callee)
                                or receiver.callee)
                    func_name = method_source.method
            shape = _match_result_python_shape(func_top, func_name)
            if shape is not None:
                return (shape.kind, shape.item_kind)

        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)):
            receiver_kind, receiver_item_kind = (
                self._expression_container_shape(node.func.value))
            method = node.func.attr
            if (receiver_kind == "dict"
                    and method == "get"
                    and receiver_item_kind):
                return (receiver_item_kind, "")
            shape = _builtin_method_return_shape(
                PythonShape(receiver_kind, receiver_item_kind), method)
            if shape is not None:
                return (shape.kind, shape.item_kind)

        return ("", "")

    ## Preserve a concrete Python value shape for local call arguments.
    #
    #  Container shapes reuse lexical flow facts. Scalar literals are carried
    #  by their builtin type so downstream method resolution can validate the
    #  protocol instead of treating every Python value as interchangeable.
    #  @param node Value expression.
    #  @return PythonShape or None.
    def _expression_python_shape(self, node):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts:
            item_shapes = [
                self._expression_python_shape(item) for item in node.elts
            ]
            if (any(item is None for item in item_shapes)
                    or any(item.kind != item_shapes[0].kind
                           for item in item_shapes[1:])):
                return None
            return PythonShape(
                _container_kind(node) or "", item_shapes[0].kind)
        kind, item_kind = self._expression_container_shape(node)
        if kind:
            return PythonShape(kind, item_kind)
        if isinstance(node, ast.Constant):
            value = node.value
            if value is None:
                return PythonShape("NoneType")
            value_type = type(value)
            if value_type in (bytes, bool, int, float, complex):
                return PythonShape(value_type.__name__)
        return None

    ## Return the container kind relevant to a method call receiver.
    #  @param node The ast.Call node.
    #  @return Container or subscript-item kind string, or None.
    def _call_receiver_container_kind(self, node):
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, ast.Attribute):
            return None
        receiver = node.func.value
        literal_kind = _container_kind(receiver)
        if literal_kind is not None:
            return literal_kind
        if isinstance(receiver, ast.Name):
            return self._lookup_container_kind(receiver.id)
        if isinstance(receiver, ast.Attribute) and self._class_stack:
            name = self._attribute_name(receiver)
            if name and name.startswith("self."):
                return self.instance_attr_kinds.get(
                    (self._class_stack[-1], name))
            if isinstance(receiver.value, ast.Name):
                root = receiver.value.id
                class_name = self._class_stack[-1]
                active_receiver = (
                    self._class_receiver_stack[-1]
                    if self._class_receiver_stack else "")
                if root == class_name or (
                        active_receiver and root == active_receiver):
                    return self.class_attr_kinds.get(
                        (class_name, receiver.attr))
        if (isinstance(receiver, ast.Subscript)
                and isinstance(receiver.value, ast.Name)):
            return self._lookup_container_kind(receiver.value.id, item=True)
        if (isinstance(receiver, ast.Subscript)
                and isinstance(receiver.value, ast.Attribute)
                and self._class_stack):
            name = self._attribute_name(receiver.value)
            if name and name.startswith("self."):
                return self.instance_attr_item_kinds.get(
                    (self._class_stack[-1], name))
        if (isinstance(receiver, ast.Subscript)
                and isinstance(receiver.value, ast.Call)):
            producer = receiver.value
            producer_kind = self._call_receiver_container_kind(producer)
            producer_method = (
                producer.func.attr
                if isinstance(producer.func, ast.Attribute) else "")
            item_kind = _BUILTIN_METHOD_RESULT_ITEM_KINDS.get(
                (producer_kind, producer_method))
            if item_kind is not None:
                return item_kind
            producer_owner = self.get_base(producer, call_lookup=True)
            if producer_owner == "python":
                return _BUILTIN_METHOD_RESULT_ITEM_KINDS.get(
                    ("str", producer_method))
        receiver_kind, _ = self._expression_container_shape(receiver)
        if receiver_kind:
            return receiver_kind
        return None

    ## Record the concrete kind assigned through a dictionary subscript.
    #
    #  The evidence is accepted only for a receiver already known to be a
    #  Python dict. All writes must converge to one concrete item kind.
    #  Conflicting or unresolved writes invalidate the fact for that binding.
    #  @param target Assignment target AST node.
    #  @param value_kind Concrete kind of the assigned value, or None.
    def _record_subscript_item_kind(self, target, value_kind):
        if (not isinstance(target, ast.Subscript)
                or not isinstance(target.value, ast.Name)):
            return
        container_name = target.value.id
        binding = self.current_scope().lookup(
            container_name, skip_parent_classes=True)
        if binding is None or binding.container_kind != "dict":
            return

        conflict_key = self._binding_key(binding)
        if conflict_key in self._container_item_kind_conflicts:
            return
        current = binding.container_item_kind or ""
        if not value_kind or (current and current != value_kind):
            binding.container_item_kind = ""
            self.container_item_kinds.pop(container_name, None)
            self._container_item_kind_conflicts.add(conflict_key)
            return
        binding.container_item_kind = value_kind
        self.container_item_kinds[container_name] = value_kind

    ## Build a scope-qualified key for a locally assigned callable.
    #  @param name Assignment target name.
    #  @return Stable key used in return_sources.
    def _local_callable_key(self, name):
        parts = list(self._class_stack) + list(self._func_stack) + [name]
        return ".".join(parts)

    ## Build a stable scope-qualified key for an inline lambda callable.
    #  @param node Lambda expression.
    #  @return Source-location-qualified callable key.
    def _local_lambda_key(self, node):
        name = "<lambda>@%d:%d" % (node.lineno, node.col_offset)
        return self._local_callable_key(name)

    ## Preserve a local callable's identity when stored as a value.
    #  @param node Value expression.
    #  @return Qualified callable key or ordinary traced source.
    def _value_source(self, node):
        if (isinstance(node, (ast.Constant, ast.JoinedStr))
                or _container_kind(node) is not None):
            return "python"
        if isinstance(node, ast.Name):
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if binding is not None and binding.callable_key:
                return binding.callable_key
        if isinstance(node, ast.Lambda):
            return self._local_lambda_key(node)
        return self.trace_source(node) or self.get_base(node)

    ## Preserve a lexical local callable's identity on a call edge.
    #
    #  Nested functions and lambdas need their qualified binding key so
    #  call contexts cannot merge unrelated same-named local definitions.
    #  @param node Call target expression.
    #  @return Local callable key or the ordinary traced source.
    def _call_edge_callee_source(self, node):
        if isinstance(node, ast.Name):
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if (binding is not None and binding.callable_key
                    and binding.callable_key != node.id):
                return binding.callable_key
        return self.trace_source(node)

    ## Return argparse destination fields tracked for a parser binding.
    #
    #  @param parser_name Parser variable name.
    #  @return Mutable destination-name set, or None.
    def _argparse_destinations(self, parser_name):
        scope = self.current_scope()
        while scope is not None:
            destinations = self._argparse_parsers.get(
                (id(scope), parser_name))
            if destinations is not None:
                return destinations
            scope = scope.parent
        return None

    ## Return Python shapes tracked for an argparse parser binding.
    #
    #  @param parser_name Parser variable name.
    #  @return Destination-to-PythonShape mapping, or None.
    def _argparse_destination_shape_map(self, parser_name):
        scope = self.current_scope()
        while scope is not None:
            shapes = self._argparse_destination_shapes.get(
                (id(scope), parser_name))
            if shapes is not None:
                return shapes
            scope = scope.parent
        return None

    ## Record an argparse add_argument() destination with Python value shape.
    #
    #  Custom type/action callables are intentionally excluded because their
    #  return object may be project-local or import-backed.
    #  @param node Candidate add_argument() call.
    def _collect_argparse_destination(self, node):
        if (not isinstance(node.func, ast.Attribute)
                or node.func.attr != "add_argument"
                or not isinstance(node.func.value, ast.Name)):
            return
        destinations = self._argparse_destinations(node.func.value.id)
        if destinations is None:
            return
        shapes = self._argparse_destination_shape_map(node.func.value.id)

        keywords = {
            keyword.arg: keyword.value for keyword in node.keywords
            if keyword.arg is not None
        }
        type_node = keywords.get("type")
        if (type_node is not None
                and (not isinstance(type_node, ast.Name)
                     or not _is_builtin(type_node.id))):
            return
        action_node = keywords.get("action")
        if (action_node is not None
                and (not isinstance(action_node, ast.Constant)
                     or not isinstance(action_node.value, str))):
            return

        destination = None
        dest_node = keywords.get("dest")
        if (isinstance(dest_node, ast.Constant)
                and isinstance(dest_node.value, str)):
            destination = dest_node.value
        else:
            option_strings = [
                arg.value for arg in node.args
                if (isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str))
            ]
            long_options = [
                option for option in option_strings
                if option.startswith("--")
            ]
            if long_options:
                destination = long_options[0][2:].replace("-", "_")
            elif option_strings and not option_strings[0].startswith("-"):
                destination = option_strings[0].replace("-", "_")
        if destination:
            destinations.add(destination)
            if shapes is not None:
                shape = self._expression_python_shape(
                    keywords.get("default"))
                if shape is None and isinstance(type_node, ast.Name):
                    if _is_builtin(type_node.id):
                        shape = PythonShape(type_node.id)
                if (shape is None
                        and isinstance(action_node, ast.Constant)
                        and action_node.value in (
                            "store_true", "store_false")):
                    shape = PythonShape("bool")
                if shape is None:
                    shapes.pop(destination, None)
                else:
                    existing = shapes.get(destination)
                    if existing is None or existing == shape:
                        shapes[destination] = shape
                    else:
                        shapes.pop(destination, None)

    ## Record ArgumentParser construction or parse_args Namespace attributes.
    #
    #  @param node Assignment being visited.
    def _collect_argparse_assignment(self, node):
        if (not isinstance(node.value, ast.Call)
                or not isinstance(node.value.func, ast.Attribute)):
            return
        targets = [
            target.id for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if not targets:
            return

        # ArgumentGroup.add_argument() populates the same Namespace as its
        # owning ArgumentParser.  Preserve that destination set when a group
        # is assigned to a local name, so grouped options receive the same
        # Python value-shape evidence as parser-level options.
        if (node.value.func.attr == "add_argument_group"
                and isinstance(node.value.func.value, ast.Name)):
            destinations = self._argparse_destinations(
                node.value.func.value.id)
            shapes = self._argparse_destination_shape_map(
                node.value.func.value.id)
            if destinations is not None:
                for target in targets:
                    self._argparse_parsers[
                        (id(self.current_scope()), target)] = destinations
                    if shapes is not None:
                        self._argparse_destination_shapes[
                            (id(self.current_scope()), target)] = shapes
                return

        func_top, func_name = self._resolve_func_top(node.value.func)
        if func_top == "argparse" and func_name == "ArgumentParser":
            for target in targets:
                self._argparse_parsers[
                    (id(self.current_scope()), target)] = set()
                self._argparse_destination_shapes[
                    (id(self.current_scope()), target)] = {}
            return

        if (node.value.func.attr not in ("parse_args",)
                or not isinstance(node.value.func.value, ast.Name)):
            return
        destinations = self._argparse_destinations(
            node.value.func.value.id)
        if destinations is None:
            return
        shapes = self._argparse_destination_shape_map(
            node.value.func.value.id)
        for target in targets:
            for destination in destinations:
                shape = shapes.get(destination) if shapes is not None else None
                self._bind_target_name(
                    target + "." + destination,
                    "python",
                    node,
                    "attribute",
                    container_kind=(shape.kind if shape is not None
                                    and shape.kind in (
                                        "list", "dict", "set", "tuple",
                                        "str") else ""),
                    container_item_kind=(shape.item_kind if shape is not None
                                         else ""),
                )

    ## Preserve a direct import-from callable for argument-flow evidence.
    #
    #  This metadata is intentionally separate from the ordinary symbol
    #  binding so public provenance and legacy call classification keep their
    #  established representation.
    #
    #  @param node Candidate call expression.
    #  @return Qualified CallResult or None.
    def _imported_call_result_source(self, node):
        if (not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)):
            return None
        qualified = self.import_from_symbols.get(node.func.id)
        if not qualified:
            return None
        traced = normalize_source(self.trace_source(node))
        return CallResult(
            qualified,
            display_name=node.func.id,
            call_lineno=node.lineno,
            call_col_offset=node.col_offset,
            result_source=(traced.result_source
                           if isinstance(traced, CallResult) else None),
        )

    ## Preserve parameter forwarding in call-edge argument facts.
    #  @param node Argument expression.
    #  @return ParameterSource or ordinary argument source.
    def _call_edge_argument_source(self, node):
        if isinstance(node, ast.Subscript):
            dependency = self._parameter_dependency_source(node.value)
            if isinstance(dependency, ParameterSource):
                if isinstance(node.slice, ast.Slice):
                    return ParameterSource(
                        dependency.scope,
                        dependency.name,
                        derived=True,
                        attributes=dependency.attributes,
                        derived_operation="slice",
                    )
                index = self._get_slice(node.slice)
                return ContainerItem(
                    dependency, index if index is not None else "*")
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.List,
                             ast.Tuple, ast.Set, ast.Dict)):
            python_shape = self._expression_python_shape(node)
            if python_shape is not None:
                return python_shape
        if isinstance(node, ast.Name) and self._caller_stack:
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if (binding is not None and isinstance(binding.source, SourceSet)
                    and binding.source.origin == "function_branch"):
                return binding.source
            if (self.current_scope().kind == SCOPE_COMPREHENSION
                    and node.id in self.comprehension_targets
                    and binding is not None
                    and (normalize_source(binding.source) == "local"
                         or (isinstance(
                             normalize_source(binding.source), CallResult)
                             and normalize_source(
                                 binding.source).result_source is None))):
                return UnknownSource("unresolved iterator element")
            if (binding is not None
                    and binding.binding_kind == "parameter"):
                return ParameterSource(
                    self._caller_stack[-1].qualname, node.id)
            assigned_call = self._assigned_call_sources.get(
                (id(self.current_scope()), node.id))
            if assigned_call is not None:
                return assigned_call
        imported_call = self._imported_call_result_source(node)
        if imported_call is not None:
            return imported_call
        return self._value_source(node)

    ## Preserve the exact producing call for a receiver-protocol argument.
    #  @param node Argument or return expression.
    #  @return Concrete Python shape, exact CallResult, or None.
    def _call_edge_protocol_source(self, node):
        if isinstance(node, ast.Name):
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if (binding is not None and isinstance(binding.source, SourceSet)
                    and binding.source.origin == "function_branch"):
                return binding.source
        if isinstance(node, ast.Call):
            if (_is_unshadowed_builtin_call(self, node)
                    and node.func.id in _BUILTIN_PYTHON_OWNED_RESULT
                    and node.func.id != "super"
                    and _builtin_shape_type(node.func.id) is not None):
                return PythonShape(node.func.id)
            source = self._call_edge_argument_source(node)
            shape = self._expression_python_shape(node)
            if (isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Subscript)):
                receiver = self._call_edge_argument_source(node.func.value)
                if (isinstance(receiver, ContainerItem)
                        and isinstance(receiver.container, ParameterSource)):
                    return CallResult(
                        source, display_name=ast.unparse(node.func),
                        call_lineno=node.lineno, call_col_offset=node.col_offset,
                        result_source=InstanceMethod(receiver, node.func.attr))
            chained = (isinstance(node.func, ast.Attribute)
                       and isinstance(node.func.value, ast.Call))
            if isinstance(source, CallResult) and not chained:
                if shape is not None:
                    return replace(source, result_source=shape)
                return source
            return CallResult(
                (source.callee if isinstance(source, CallResult) else source)
                or UnknownSource("unresolved return callable"),
                display_name=ast.unparse(node.func),
                call_lineno=node.lineno,
                call_col_offset=node.col_offset,
                result_source=shape or source)
        return self._expression_python_shape(node)

    ## Preserve exact fields of a callback argument tuple.
    #  @param node Candidate tuple/list expression.
    #  @return TupleSource with UnknownSource fields when needed, or None for
    #  a non-tuple expression.
    def _call_edge_tuple_source(self, node):
        if not isinstance(node, (ast.Tuple, ast.List)):
            return None
        fields = []
        for field in node.elts:
            source = normalize_source(self._call_edge_argument_source(field))
            fields.append(source or UnknownSource("unresolved callback argument"))
        return TupleSource(tuple(fields))

    ## Attach the defining module to a declaration-time call source.
    #  Default expressions are evaluated while visiting a function definition,
    #  outside the function call graph. The module fact keeps alias resolution
    #  available when that default is later propagated to a receiver.
    #  @param source Candidate argument source.
    #  @return Source with module metadata when it is a CallResult.
    def _default_argument_source(self, source):
        return self._source_with_module(source)

    ## Attach this analyzer's module to a call result when absent.
    #  @param source Candidate source value.
    #  @return Source with module metadata when applicable.
    def _source_with_module(self, source):
        source = normalize_source(source)
        if not isinstance(source, CallResult) or source.source_module:
            return source
        return CallResult(
            source.callee,
            display_name=source.display_name,
            call_lineno=source.call_lineno,
            call_col_offset=source.call_col_offset,
            source_module=self.module_name or "",
            result_source=source.result_source,
        )

    ## Return a project-local callback identity carried by one argument.
    #  @param node Call argument AST node.
    #  @return Local function or qualified self-method name, otherwise None.
    def _call_edge_callback_name(self, node):
        if isinstance(node, ast.Name):
            if (node.id in self.defined_functions
                    or node.id in self.class_methods):
                return node.id
            return None
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in ("self", "cls")
                and self._class_stack
                and node.attr in self.class_methods.get(
                    self._class_stack[-1], [])):
            return self._class_stack[-1] + "." + node.attr
        return None

    ## Preserve element ownership for a call argument used as an iterable.
    #
    #  This fact is separate from ordinary argument ownership. A Python list
    #  may yield project-local or import-backed objects, so treating the list
    #  owner as the element owner would contaminate receiver classification.
    #  @param node Argument expression.
    #  @return Element source or None when the iterable is not explicit.
    def _call_edge_iterable_source(self, node):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if not node.elts:
                return UnknownSource("empty iterable")
            return _iterable_element_source(node, self.trace_source)
        if isinstance(node, ast.Name):
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if binding is not None:
                tuple_sources = self._iterated_append_tuple_sources.get(
                    self._binding_key(binding))
                if tuple_sources is not None:
                    return tuple_sources
                source = self._iterable_binding_sources.get(
                    self._binding_key(binding))
                if source is not None:
                    return source
                dependency = self._parameter_dependency_source(node)
                if dependency is not None:
                    return ContainerIter(dependency)
        return None

    ## Attach an explicit iterable's element source to its lexical binding.
    #
    #  Binding identity prevents a same-name container in another function
    #  from changing an already collected call edge.
    #  @param node Assignment node whose targets have already been bound.
    def _record_iterable_binding_source(self, node):
        if not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            return
        if not node.value.elts:
            item_source = UnknownSource("empty iterable")
        else:
            item_source = _iterable_element_source(
                node.value, self.trace_source)
        if item_source is None:
            return
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            binding = self.current_scope().bindings.get(target.id)
            if binding is not None:
                self._iterable_binding_sources[
                    self._binding_key(binding)] = item_source

    ## Resolve the result object of a builtin container method.
    #
    #  The method callable is Python-owned, but selection methods return a
    #  value from the receiver. Preserve that item source instead of
    #  contaminating the result object with the callable's owner.
    #  @param node Method call.
    #  @param receiver_kind Independently known builtin container kind.
    #  @param method_name Called method name.
    #  @return Result source, "python", or None.
    def _builtin_method_result_source(
            self, node, receiver_kind, method_name):
        if (receiver_kind == "dict"
                and method_name == "get"
                and node.args):
            receiver = node.func.value
            if isinstance(receiver, ast.Name):
                container = receiver.id
            else:
                container = self.trace_source(receiver)
            key_node = node.args[0]
            if isinstance(key_node, ast.Constant):
                key = key_node.value
            else:
                key = "*"
            selected = ContainerItem(container, key)
            if len(node.args) >= 2:
                default = _builtin_value_source(
                    node.args[1], self.trace_source)
                return make_source_set(
                    (selected, default), origin="dict_lookup")
            return selected
        return "python"

    ## Preserve whether an assignment value has unresolved parameter origin.
    #  @param node Assignment value expression.
    #  @return ParameterSource, DerivedResult, UnknownSource, "local", or
    #  None.
    def _parameter_dependency_source(self, node, expression_context=False):
        if isinstance(node, ast.Name):
            if not self._caller_stack:
                return None
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if binding is None:
                return None
            existing = normalize_source(binding.source)
            if isinstance(existing, (ParameterSource, UnknownSource)):
                return existing
            if isinstance(existing, DerivedResult):
                return existing
            if (expression_context
                    and isinstance(existing, ContainerItem)):
                return UnknownSource(
                    "unresolved container-item expression")
            if (expression_context
                    and isinstance(existing, CallResult)
                    and existing.result_source is None):
                callee = existing.callee
                callee_root = (
                    callee.split(".", 1)[0]
                    if isinstance(callee, str) else "")
                imported = (
                    callee_root in self.import_aliases
                    or callee_root in self.import_from_symbols
                    or (callee_root in self.symbols.direct
                        and self.symbols.direct.get(callee_root)
                        not in (None, "local", "python", "unknown")))
                if not imported:
                    return None
                # A call result without an explicit result-owner contract
                # cannot be safely used as the operand owner of a later
                # expression. Preserve uncertainty instead of allowing the
                # assignment fallback to relabel the value as local.
                return UnknownSource("unresolved call-result expression")
            if binding.binding_kind != "parameter":
                return None
            return ParameterSource(
                self._caller_stack[-1].qualname, node.id)

        if isinstance(node, ast.Attribute):
            if (expression_context
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("self", "cls")):
                attr_name = self._attribute_name(node)
                existing_attr = None
                if self._class_stack and attr_name:
                    existing_attr = self.instance_attrs.get(
                        (self._class_stack[-1], attr_name))
                existing_attr = normalize_source(existing_attr)
                if existing_attr is not None:
                    if isinstance(existing_attr, (ParameterSource,
                                                   UnknownSource,
                                                   DerivedResult)):
                        return existing_attr
                    return None
                scope_name = (
                    self._caller_stack[-1].qualname
                    if self._caller_stack else "")
                if not self._class_stack:
                    return "local"
                return InstanceAttribute(
                    self._class_stack[-1], attr_name, scope_name)
            dependency = self._parameter_dependency_source(
                node.value, expression_context=expression_context)
            if isinstance(dependency, ParameterSource):
                return ParameterSource(
                    dependency.scope,
                    dependency.name,
                    derived=dependency.derived,
                    attributes=dependency.attributes + (node.attr,),
                    derived_operation=dependency.derived_operation,
                )
            if isinstance(dependency, UnknownSource):
                return dependency
            return None

        if isinstance(node, ast.Subscript):
            dependency = self._parameter_dependency_source(
                node.value, expression_context=expression_context)
            if isinstance(dependency, ParameterSource):
                return ParameterSource(
                    dependency.scope,
                    dependency.name,
                    derived=True,
                    attributes=dependency.attributes,
                    derived_operation=(
                        "slice" if isinstance(node.slice, ast.Slice)
                        else "item"),
                )
            if isinstance(dependency, UnknownSource):
                return dependency
            return None

        if isinstance(node, ast.UnaryOp):
            dependency = self._parameter_dependency_source(
                node.operand, expression_context=expression_context)
            if dependency is not None:
                return UnknownSource("unresolved parameter-derived expression")
            return None

        if isinstance(node, ast.BinOp):
            left = self._parameter_dependency_source(
                node.left, expression_context=True)
            right = self._parameter_dependency_source(
                node.right, expression_context=True)
            if left is not None or right is not None:
                if left is None:
                    traced_left = self.trace_source(node.left)
                    if traced_left != "local":
                        left = traced_left
                if right is None:
                    traced_right = self.trace_source(node.right)
                    if traced_right != "local":
                        right = traced_right
                operands = tuple(
                    source for source in (left, right)
                    if source is not None)
                return DerivedResult(
                    "expression", operands,
                    type(node.op).__name__)
            return None

        if isinstance(node, ast.Compare):
            operands = []
            has_dependency = False
            for operand in [node.left] + list(node.comparators):
                dependency = self._parameter_dependency_source(
                    operand, expression_context=True)
                if dependency is not None:
                    has_dependency = True
                    operands.append(dependency)
                    continue
                traced = self.trace_source(operand)
                if traced != "local" and traced is not None:
                    operands.append(traced)
            if has_dependency:
                return DerivedResult(
                    "expression", tuple(operands), "Compare")
            return None

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            dependency = self._parameter_dependency_source(node.func.value)
            if dependency is not None:
                method_source = self._resolve_methods(node)
                if method_source is None:
                    return UnknownSource("unresolved parameter method result")
                return CallResult(
                    method_source,
                    display_name=ast.unparse(node.func),
                    call_lineno=node.lineno,
                    call_col_offset=node.col_offset,
                    result_source=DerivedResult(
                        "method_result",
                        (method_source,),
                        node.func.attr,
                    ),
                )
        return None

    ## Record a SymbolRef for provenance tracking.
    #  @param symbol Display name.
    #  @param source Source value.
    #  @param kind Symbol category.
    #  @param node Optional AST node for position.
    def _add_symbol_ref(self, symbol, source, kind, node=None):
        scope_name = ""
        cs = self.current_scope()
        if cs.kind != SCOPE_MODULE:
            scope_name = cs.name
        self.symbol_refs.append(SymbolRef(
            symbol=symbol,
            source=source,
            kind=kind,
            module_name=self.module_name or "",
            file_path=getattr(self, '_file_path', ""),
            scope_name=scope_name,
            lineno=getattr(node, "lineno", 0) if node is not None else 0,
            col_offset=getattr(node, "col_offset", 0) if node is not None else 0,
        ))

    ## --- Import visitors ---

    ## Visit an Import node and record alias-to-module mappings.
    #  @param node The Import AST node.
    def visit_Import(self, node):
        for alias in node.names:
            symbol = alias.asname if alias.asname else alias.name
            self.import_aliases.add(symbol)
            self._bind_target_name(symbol, alias.name, node, "import")
            binding = self.current_scope().bindings.get(symbol)
            if binding is not None:
                self._import_binding_sources[
                    self._binding_key(binding)] = alias.name
        self.generic_visit(node)

    ## Visit an ImportFrom node and record alias-to-module mappings.
    #  @param node The ImportFrom AST node.
    def visit_ImportFrom(self, node):
        for alias in node.names:
            symbol = alias.asname if alias.asname else alias.name
            if symbol == '*':
                if node.module:
                    if node.level > 0 and self.module_name:
                        resolved = self._resolve_relative_import(node.module, node.level)
                        self.wildcard_modules.append(resolved)
                    else:
                        self.wildcard_modules.append(node.module)
                continue
            if node.level > 0 and self.module_name:
                resolved = self._resolve_relative_import(node.module, node.level)
                self._bind_target_name(symbol, resolved, node, "import")
                qualified = (
                    (resolved + '.' + alias.name) if resolved else alias.name)
                self.import_from_symbols[symbol] = qualified
            else:
                self.import_aliases.add(symbol)
                self._bind_target_name(symbol, node.module, node, "import")
                qualified = (
                    (node.module + '.' + alias.name)
                    if node.module else alias.name)
                self.import_from_symbols[symbol] = qualified
            binding = self.current_scope().bindings.get(symbol)
            if binding is not None:
                self._import_binding_sources[
                    self._binding_key(binding)] = qualified
        self.generic_visit(node)

    ## Resolve a relative import to its full dotted module name.
    #  @param module The module portion after the dots (may be None for "from . import X").
    #  @param level The number of leading dots (1 = current package, 2 = parent, etc.).
    #  @return The full dotted module name.
    def _resolve_relative_import(self, module, level):
        if not self.module_name:
            return module or ''
        parts = self.module_name.split('.')
        ## __package__: for packages use module_name, else use parent
        if self.is_package:
            pkg_parts = parts
        else:
            if len(parts) < 2:
                return module or ''
            pkg_parts = parts[:-1]
        ## level dots = go up (level-1) from __package__
        strip = level - 1
        if strip >= len(pkg_parts):
            base = ''
        elif strip == 0:
            base = '.'.join(pkg_parts)
        else:
            base = '.'.join(pkg_parts[:-strip])
        if module:
            return f"{base}.{module}" if base else module
        return base

    ## --- Source tracing ---

    ## Trace an AST expression node back to its source symbol/structured origin.
    #
    #  Handles Name, Call, Attribute, Lambda, Subscript, and literal nodes.
    #  For Call nodes, tries getattr(), importlib.import_module(), partial(),
    #  method resolution, and chained-call receiver resolution.
    #  @param node The AST expression node.
    #  @return A symbol string, a structured tuple, or None.
    def trace_source(self, node):
        if isinstance(node, ast.Name):
            return self._lookup_name_source(node.id)
        elif isinstance(node, ast.Call):
            getattr_src = self._resolve_getattr_trace(node)
            selected = self._finite_eval_names(node)
            if selected is not None:
                return selected
            if getattr_src:
                return getattr_src
            import_mod = self._resolve_import_module_trace(node)
            if import_mod:
                return import_mod
            if self._is_partial_call(node) and node.args:
                return self.get_base(node.args[0])
            me = self._resolve_methods(node)
            if me:
                receiver_kind = self._call_receiver_container_kind(node)
                if (receiver_kind is not None
                        and isinstance(me, InstanceMethod)
                        and me.method in _BUILTIN_CONTAINER_METHODS.get(
                            receiver_kind, frozenset())):
                    return CallResult(
                        me,
                        display_name=ast.unparse(node.func),
                        call_lineno=node.lineno,
                        call_col_offset=node.col_offset,
                        result_source=self._builtin_method_result_source(
                            node, receiver_kind, me.method),
                    )
                if (isinstance(me, InstanceMethod)
                        and me.receiver == "python"):
                    return CallResult(
                        me,
                        display_name=ast.unparse(node.func),
                        call_lineno=node.lineno,
                        call_col_offset=node.col_offset,
                        result_source="python",
                    )
                ## 7B-full PR5: if a local class method has an import-backed
                ## return source, propagate it so assigned variables carry
                ## library provenance.
                if (isinstance(me, InstanceMethod) and isinstance(me.method, str)
                        and isinstance(me.receiver, str)):
                    # Check whether the receiver is a local class instance.
                    local_class = me.receiver in self.class_methods
                    if not local_class:
                        binding = self.current_scope().lookup(me.receiver)
                        if binding is not None:
                            src = normalize_source(binding.source)
                            if isinstance(src, CallResult) and isinstance(src.callee, str):
                                local_class = src.callee in self.class_methods
                    if local_class:
                        # Determine the qualname for return_sources lookup.
                        class_name = me.receiver
                        if not (me.receiver in self.class_methods):
                            binding = self.current_scope().lookup(me.receiver)
                            if binding is not None:
                                src = normalize_source(binding.source)
                                if isinstance(src, CallResult) and isinstance(src.callee, str):
                                    class_name = src.callee
                        method_key = class_name + "." + me.method
                        # Keep the local callable identity instead of choosing
                        # one return branch during the single-file pass.
                        # Cross-file resolution evaluates the completed return
                        # summary after the whole module has been visited.
                        return CallResult(
                            method_key,
                            display_name=ast.unparse(node.func),
                            call_lineno=node.lineno,
                            call_col_offset=node.col_offset,
                        )
                ## 1.0.5 P2: SuperMethod identifies the call target;
                #  wrap in CallResult so the return value does not inherit
                #  the base-class owner.  result_source=UnknownSource:
                #  without return-type evidence, result is unknowable.
                if isinstance(me, SuperMethod):
                    return CallResult(me,
                                      display_name="super().%s" % me.method,
                                      call_lineno=node.lineno,
                                      call_col_offset=node.col_offset,
                                      result_source=UnknownSource("super()"))
                return me
            ## For chained calls (A().B()), resolve via the inner call's
            ## return source so the outer call traces to the correct library.
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
                inner_source = self.trace_source(node.func.value)
                if isinstance(inner_source, str):
                    rs = self.return_sources.get(inner_source)
                    if rs is not None:
                        rs = normalize_source(rs)
                        if isinstance(rs, SourceSet):
                            inner_source = rs
                        else:
                            return rs
                if isinstance(inner_source, CallResult):
                    if inner_source.result_source is not None:
                        result_owner = inner_source.result_source
                        if (isinstance(result_owner, str)
                                and isinstance(node.func, ast.Attribute)):
                            mapped_owner = _match_result_owner(
                                result_owner, node.func.attr)
                            if mapped_owner is not None:
                                mapped_shape = _match_result_python_shape(
                                    result_owner, node.func.attr)
                                return CallResult(
                                    InstanceMethod(
                                        result_owner, node.func.attr),
                                    display_name=ast.unparse(node.func),
                                    call_lineno=node.lineno,
                                    call_col_offset=node.col_offset,
                                    result_source=(
                                        mapped_shape or mapped_owner))
                        return result_owner
                    rs = self.return_sources.get(inner_source.callee)
                    if rs is not None:
                        rs = normalize_source(rs)
                        if isinstance(rs, SourceSet):
                            inner_source = rs
                        else:
                            return rs
                if isinstance(inner_source, SourceSet):
                    return inner_source
                if inner_source:
                    return inner_source
            if isinstance(node.func, ast.Name):
                binding = self.current_scope().lookup(
                    node.func.id, skip_parent_classes=True)
                if binding is not None and binding.callable_key:
                    call_key = binding.callable_key
                elif (node.func.id in self.defined_functions
                      or node.func.id in self.class_methods):
                    call_key = node.func.id
                else:
                    call_key = self.get_base(node, call_lookup=True)
            else:
                call_key = self.get_base(node, call_lookup=True)
            method_source = normalize_source(self._resolve_methods(node))
            if (isinstance(method_source, InstanceMethod)
                    and isinstance(method_source.receiver, str)
                    and _is_verified_result_owner(
                        method_source.receiver)):
                call_key = method_source.receiver
            if call_key:
                if isinstance(call_key, CallResult):
                    return call_key
                ## Resolve self.attr through instance_attrs so that
                ## chained calls propagate library provenance, e.g.
                ## predictions = self.model.predict(X)[0].reshape(...)
                ## where self.model -> GPy.models.GPRegression.
                if isinstance(call_key, str) and call_key.startswith("self.") and self._class_stack:
                    cn = self._class_stack[-1]
                    attr_src = self.instance_attrs.get((cn, call_key))
                    if isinstance(attr_src, CallResult) and isinstance(attr_src.callee, str):
                        call_key = attr_src.callee
                ## Extract import-backed receiver from InstanceMethod.
                if isinstance(call_key, InstanceMethod) and isinstance(call_key.receiver, str):
                    call_key = call_key.receiver
                display = ""
                try:
                    display = ast.unparse(node.func)
                except Exception:
                    pass
                if isinstance(call_key, str) and '.' not in display:
                    display = ""
                ## 1.0.5 P2: determine result-object ownership for builtin callees.
                rs = None
                if isinstance(call_key, str):
                    local_returns = self.return_sources.get(call_key)
                    local_shape = _uniform_python_shape(local_returns)
                    if local_shape is not None:
                        rs = local_shape
                    elif _is_uniform_python_result(local_returns):
                        rs = "python"
                    elif _has_python_result(local_returns):
                        rs = UnknownSource("mixed local return")
                    func_top, func_name = self._resolve_func_top(node.func)
                    mapped_owner = _match_result_owner(func_top, func_name)
                    if mapped_owner is not None:
                        rs = (
                            _match_result_python_shape(func_top, func_name)
                            or mapped_owner)
                    else:
                        preserved = self._receiver_preserving_result_owner(
                            node, func_top, func_name)
                        if preserved is not None:
                            rs = preserved
                    if _is_unshadowed_builtin_call(self, node):
                        rs = _resolve_builtin_result(
                            call_key, node, self.trace_source)
                return CallResult(call_key, display_name=display,
                                  call_lineno=node.lineno,
                                  call_col_offset=node.col_offset,
                                  result_source=rs)
            return self.get_base(node.func)
        elif isinstance(node, ast.Attribute):
            name = self._attribute_name(node)
            if name and name in self.symbols.direct:
                return name
            return self.get_base(node)
        elif isinstance(node, ast.Lambda):
            body_base = self.get_base(node.body)
            if isinstance(body_base, str):
                param_names = {a.arg for a in node.args.args}
                if node.args.vararg:
                    param_names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    param_names.add(node.args.kwarg.arg)
                if body_base in param_names:
                    return "local"
            return body_base
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Call):
                item_owner = self._resolve_call_result_item_owner(node.value)
                if item_owner is not None:
                    return item_owner
            container_name = self.trace_source(node.value)
            key_idx = self._get_slice(node.slice)
            if container_name is not None and key_idx is not None:
                # Use the variable *name* for the container_items lookup
                # (container_items is keyed by name, not by trace source).
                lookup_name = node.value.id if isinstance(node.value, ast.Name) else container_name
                key_value = self._container_index(lookup_name, key_idx)
                lookup_key = (lookup_name, key_value)
                if lookup_key in self.container_items:
                    return self.container_items[lookup_key]
                return ContainerItem(lookup_name, key_idx)
            ## 7B-full PR7: try to resolve static key from literal assignment.
            resolved_key = None
            if isinstance(node.value, ast.Name):
                var_name = node.value.id
                if isinstance(node.slice, ast.Name):
                    resolved_key = self._literal_values.get(node.slice.id)
                if resolved_key is not None:
                    lookup = self.container_items.get((var_name, resolved_key))
                    if lookup is not None:
                        return lookup
            if isinstance(node.value, ast.Name):
                homogeneous = self.homogeneous_container_value_sources.get(
                    node.value.id)
                if homogeneous is not None:
                    return homogeneous
            ## Fallback: collect item sources; only create SourceSet if
            ## all candidates are consistent (single-source or same library).
            if container_name is not None and isinstance(node.value, ast.Name):
                var_name = node.value.id
                item_sources = []
                for (cn, _), src in self.container_items.items():
                    if cn == var_name:
                        item_sources.append(src)
                if item_sources:
                    return make_source_set(item_sources, origin="dict_lookup")
            return container_name
        elif isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            if isinstance(node, ast.Dict):
                value_nodes = node.values
            else:
                value_nodes = node.elts
            bases = set()
            for v in value_nodes:
                base = self.get_base(v)
                if base:
                    bases.add(base)
            if len(bases) == 1:
                return next(iter(bases))
            return None
        elif isinstance(node, ast.Constant):
            return None
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            # The value of a yield expression is supplied by generator.send()
            # or an external coroutine scheduler. It is not the value being
            # yielded, so project source does not establish its owner.
            return UnknownSource("yield expression result")
        return None

    ## Extract a string literal from an AST node.
    #  @param node The AST node.
    #  @return String value, or None.
    def _literal_str(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    ## Check if a call node is a functools.partial() call.
    #  @param node The Call AST node.
    #  @return True if the call is partial().
    def _is_partial_call(self, node):
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name) and func.id == 'partial':
            return True
        if isinstance(func, ast.Attribute) and func.attr == 'partial':
            return True
        return False

    ## Check if a call node is a getattr() call.
    #  @param node The Call AST node.
    #  @return True if the call is getattr().
    def _is_getattr_call(self, node):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            return False
        func = node.func
        if isinstance(func, ast.Name) and func.id == "getattr":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "getattr":
            return True
        return False

    ## Resolve a getattr(obj, name) call to the object's base.
    #  @param node The Call AST node.
    #  @return The object's source, or None.
    def _resolve_getattr_trace(self, node):
        if not self._is_getattr_call(node):
            return None
        name_lit = self._literal_str(node.args[1])
        if name_lit is None:
            return None
        obj_key = self.trace_source(node.args[0])
        if obj_key is None:
            return None
        return obj_key

    ## Check if a symbol ultimately originates from importlib.
    #  @param symbol The symbol to check.
    #  @return True if the symbol traces to importlib.
    def _is_importlib_module(self, symbol):
        if not isinstance(symbol, str):
            return False
        if symbol == "importlib":
            return True
        top = self.symbols.get_top(symbol)
        return top == "importlib"

    ## Check if a call node is importlib.import_module().
    #  @param node The Call AST node.
    #  @return True if the call is import_module().
    def _is_import_module_call(self, node):
        if not isinstance(node, ast.Call) or not node.args:
            return False
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "import_module":
            root = self.get_base(func.value)
            if root and self._is_importlib_module(root):
                return True
            return False
        if isinstance(func, ast.Name) and func.id == "import_module":
            if (self.import_from_symbols.get("import_module") or "").startswith("importlib"):
                return True
            return False
        return False

    ## Resolve an importlib.import_module("name") call to the module name.
    #  @param node The Call AST node.
    #  @return The module name string, or None.
    def _resolve_import_module_trace(self, node):
        if not self._is_import_module_call(node):
            return None
        name = self._literal_str(node.args[0])
        if name is None:
            return None
        return name

    ## Extract a constant integer or negated constant from a slice node.
    #  @param slice_node The AST slice node.
    #  @return Integer value, or None.
    def _get_slice(self, slice_node):
        if isinstance(slice_node, ast.Constant):
            return slice_node.value
        if isinstance(slice_node, ast.UnaryOp) and isinstance(slice_node.op, ast.USub) and isinstance(slice_node.operand, ast.Constant):
            return -slice_node.operand.value
        return None

    ## Normalize a negative container index to its positive equivalent.
    #  @param container_name The name of the container variable.
    #  @param idx The raw index value.
    #  @return Adjusted index value.
    def _container_index(self, container_name, idx):
        if not isinstance(idx, int):
            return idx
        if idx >= 0:
            return idx
        n = self.container_lengths.get(container_name)
        if n:
            return idx + n
        return idx

    ## --- Method resolution ---

    ## Resolve a method inherited from a statically known builtin base class.
    #
    #  Follows local base classes in declared MRO order.  A local override wins;
    #  an external or otherwise unknown base stops inference so a later builtin
    #  base cannot be claimed speculatively.
    #  @param class_name Local class whose bases should be inspected.
    #  @param method_name Method looked up on the instance.
    #  @param seen Local classes already visited during recursive lookup.
    #  @return Builtin type name, "local", or None when unresolved.
    def _inherited_builtin_method_owner(self, class_name, method_name,
                                        seen=None):
        if seen is None:
            seen = set()
        if class_name in seen:
            return None
        seen = set(seen)
        seen.add(class_name)

        for base in self.class_bases.get(class_name, []):
            if base in self.class_methods:
                if method_name in self.class_methods.get(base, []):
                    return "local"
                inherited = self._inherited_builtin_method_owner(
                    base, method_name, seen)
                if inherited is not None:
                    return inherited
                continue

            direct = normalize_source(self.symbols.direct.get(base))
            if direct == "local":
                return None

            imported = self.import_from_symbols.get(base, "")
            resolved_base = imported or base
            builtin_name = resolved_base.rsplit(".", 1)[-1]
            builtin_origin = (
                resolved_base == builtin_name
                or resolved_base.startswith("builtins."))
            if (builtin_origin
                    and method_name in _BUILTIN_CONTAINER_METHODS.get(
                        builtin_name, frozenset())):
                return builtin_name
            if builtin_name == "object" and builtin_origin:
                continue

            # An earlier unknown/external base may provide the descriptor.
            return None
        return None

    ## Attempt to resolve an instance method call to a class member.
    #
    #  Handles self.method(), known_object.method(), and chained attribute calls.
    #  @param node The Call AST node.
    #  @return Method name, structured ("instance_method", ...) tuple, or None.
    def _resolve_methods(self, node):
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        re = func.value
        method_name = func.attr
        # Preserve a parameter-derived expression as the receiver source.
        # Cross-file resolution may prove a single owner from all operands;
        # without this structured source, get_base() falls back to the first
        # operand and can incorrectly classify the call as local.
        if isinstance(re, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            parameter_dependency = self._parameter_dependency_source(re)
            def contains_bounded_source(source):
                source = normalize_source(source)
                if isinstance(source, (ParameterSource, InstanceAttribute)):
                    return True
                if isinstance(source, DerivedResult):
                    return any(contains_bounded_source(item)
                               for item in source.sources)
                if isinstance(source, CallResult):
                    return contains_bounded_source(source.result_source)
                return False

            if (isinstance(parameter_dependency, DerivedResult)
                    and contains_bounded_source(parameter_dependency)):
                return InstanceMethod(parameter_dependency, method_name)
            if isinstance(re, (ast.BinOp, ast.UnaryOp)):
                expression_top = self._expr_receiver_top(re)
                if expression_top not in (
                        None, "", "local", "python", "unknown"):
                    return InstanceMethod(expression_top, method_name)
                external_tops = {
                    top for top in self._operator_operand_tops(re)
                    if top not in (None, "", "local", "python", "unknown")
                }
                if external_tops or parameter_dependency is not None:
                    return InstanceMethod(
                        UnknownSource("unresolved operator result owner"),
                        method_name,
                    )
        receiver_kind, _ = self._expression_container_shape(re)
        if receiver_kind:
            if _has_builtin_shape_method(receiver_kind, method_name):
                return InstanceMethod("python", method_name)
            return InstanceMethod(
                UnknownSource(
                    "unsupported %s protocol" % receiver_kind),
                method_name,
            )

        ## 1.0.5 P2: super().method() — capture enclosing class context
        #  while _class_stack is still available during AST visit.
        if isinstance(re, ast.Call):
            if isinstance(re.func, ast.Name) and re.func.id == "super":
                if self._class_stack:
                    class_key = self._class_stack[-1]
                    class_qualname = ".".join(self._class_stack)
                    return SuperMethod(class_key, class_qualname, method_name)
                return InstanceMethod("super", method_name)

            if (isinstance(re.func, ast.Name)
                    and method_name in self.class_methods.get(
                        re.func.id, ())):
                return InstanceMethod(re.func.id, method_name)

            inner_result = normalize_source(self.trace_source(re))
            if isinstance(inner_result, CallResult):
                inner_owner = normalize_source(inner_result.result_source)
                if isinstance(inner_owner, str):
                    return InstanceMethod(inner_owner, method_name)
                if isinstance(inner_owner, UnknownSource):
                    return InstanceMethod(inner_owner, method_name)

            explicit_owner = self._explicit_external_receiver_top(re)
            if explicit_owner not in (None, "", "local", "python", "unknown"):
                return InstanceMethod(explicit_owner, method_name)

            # A direct chained method belongs to the object explicitly
            # returned by a project-local method.  Gate this path on local
            # method identity so direct library calls such as np.log(...)
            # continue through receiver-preserving return rules below.
            inner_method = self._resolve_methods(re)
            if (isinstance(inner_method, InstanceMethod)
                    and isinstance(
                        normalize_source(inner_method.receiver),
                        UnknownSource)):
                return InstanceMethod(
                    inner_method.receiver, method_name)
            local_method = False
            if (isinstance(inner_method, InstanceMethod)
                    and isinstance(inner_method.receiver, str)):
                local_method = inner_method.receiver in self.class_methods
                if not local_method:
                    binding = self.current_scope().lookup(
                        inner_method.receiver)
                    if binding is not None:
                        binding_source = normalize_source(binding.source)
                        if (isinstance(binding_source, CallResult)
                                and isinstance(binding_source.callee, str)):
                            local_method = (
                                binding_source.callee in self.class_methods)
            if local_method:
                result_source = normalize_source(self.trace_source(re))
                if isinstance(result_source, CallResult):
                    if (isinstance(result_source.callee, str)
                            and result_source.callee in self.return_sources):
                        return InstanceMethod(result_source, method_name)
                    candidate = result_source.result_source
                    if not isinstance(candidate, str):
                        candidate = result_source.callee
                    if isinstance(candidate, str):
                        candidate_top = self.symbols.get_top(candidate)
                        if candidate_top not in (
                                None, "", "local", "python", "unknown"):
                            return InstanceMethod(candidate, method_name)
            if (isinstance(inner_method, InstanceMethod)
                    and _has_result_owner_contract(
                        inner_method.method)):
                result = CallResult(
                    inner_method,
                    display_name=ast.unparse(re.func),
                    call_lineno=re.lineno,
                    call_col_offset=re.col_offset,
                    result_source=DerivedResult(
                        "method_result",
                        (inner_method,),
                        inner_method.method,
                    ),
                )
                return InstanceMethod(result, method_name)

        def _lookup_instance_attr(attr_name):
            if self._class_stack:
                class_name = self._class_stack[-1]
                if (class_name, attr_name) in self.instance_attrs:
                    return self.instance_attrs[(class_name, attr_name)]
            return self.symbols.direct.get(attr_name)

        def _resolve_on_class(class_name, receiver_key):
            if not class_name:
                return None
            methods = self.class_methods.get(class_name, [])
            if methods and method_name in methods:
                return InstanceMethod(receiver_key, method_name)
            inherited_owner = self._inherited_builtin_method_owner(
                class_name, method_name)
            if inherited_owner == "local":
                return InstanceMethod(receiver_key, method_name)
            if inherited_owner is not None:
                return InstanceMethod(inherited_owner, method_name)
            if class_name in self.class_methods:
                return InstanceMethod(receiver_key, method_name)
            if class_name in self.import_from_symbols:
                return InstanceMethod(class_name, method_name)
            return None

        def _parameter_method(receiver, parameter_name):
            if not self._caller_stack:
                return None
            binding = self.current_scope().lookup(
                parameter_name, skip_parent_classes=True)
            if (binding is None
                    or binding.binding_kind != "parameter"):
                return None
            scope_name = self._caller_stack[-1].qualname
            params = self.function_params.get(scope_name)
            if params is None:
                params = self.function_params.get(
                    scope_name.rsplit(".", 1)[-1], [])
            if parameter_name not in params:
                return None
            return InstanceMethod(
                receiver,
                method_name,
                parameter_scope=scope_name,
                parameter_name=parameter_name,
            )

        if isinstance(re, ast.Name):
            for guards in reversed(self._receiver_owner_guards):
                guarded_owner = guards.get(re.id)
                if guarded_owner is not None:
                    if isinstance(guarded_owner, PythonShape):
                        if _has_builtin_shape_method(
                                guarded_owner.kind, method_name):
                            return InstanceMethod("python", method_name)
                        return InstanceMethod(
                            UnknownSource(
                                "unsupported %s protocol"
                                % guarded_owner.kind),
                            method_name,
                        )
                    return InstanceMethod(guarded_owner, method_name)
            if re.id == "self" and self._class_stack:
                cn = self._class_stack[-1]
                return _resolve_on_class(cn, cn)
            # Lexical parameter identity takes precedence over the legacy
            # module compatibility table.  A same-named assignment in
            # another scope must not erase the call edge that can establish
            # this receiver's owner.
            parameter = _parameter_method(re.id, re.id)
            if parameter is not None:
                return parameter
            class_name = self.symbols.direct.get(re.id)
            class_name = normalize_source(class_name)
            if isinstance(class_name, CallResult):
                if isinstance(
                        normalize_source(class_name.result_source),
                        (str, UnknownSource)):
                    return InstanceMethod(class_name, method_name)
                class_name = class_name.callee
            # 1.0.5 P1: treat "local" class_name the same as
            # absent — still create InstanceMethod so the
            # builtin container method check applies.
            if not class_name or class_name == "local":
                binding = self.current_scope().lookup(re.id)
                if binding is not None:
                    src_norm = normalize_source(binding.source)
                    if isinstance(src_norm, ParameterSource):
                        if src_norm.derived:
                            return InstanceMethod(src_norm, method_name)
                        else:
                            receiver = ".".join(
                                (src_norm.name,) + src_norm.attributes)
                        return InstanceMethod(
                            receiver, method_name,
                            parameter_scope=src_norm.scope,
                            parameter_name=src_norm.name)
                    if (isinstance(src_norm, str)
                            and src_norm in
                            self.homogeneous_container_value_sources):
                        return InstanceMethod(
                            self.homogeneous_container_value_sources[
                                src_norm], method_name)
                    if binding.source == "local":
                        return InstanceMethod(re.id, method_name)
                    if isinstance(src_norm, CallResult):
                        cn = src_norm.callee
                        if isinstance(cn, str) and cn in self.class_methods:
                            return _resolve_on_class(cn, cn)
                        if isinstance(cn, str) and cn in self.import_from_symbols:
                            return InstanceMethod(cn, method_name)
                        return InstanceMethod(src_norm, method_name)
                    if is_structured_source(src_norm):
                        return InstanceMethod(src_norm, method_name)
                    if (isinstance(src_norm, str)
                            and binding.binding_kind != "import"
                            and src_norm not in (
                                "", "local", "python", "unknown")):
                        return InstanceMethod(src_norm, method_name)
                # Module-level local bindings — scope lookup
                # may return None at module scope; fall back
                # to the direct symbol table.
                if self.symbols.direct.get(re.id) == "local":
                    return InstanceMethod(re.id, method_name)
                return None
            return _resolve_on_class(class_name, re.id)

        # Preserve a parameter receiver when an item is selected before the
        # method call, e.g. values[0].reshape(...).  The item may have a
        # different runtime type from the container, so retain the parameter
        # edge and let cross_file resolve its unique call-site owner.
        if isinstance(re, ast.Subscript):
            container_dependency = self._parameter_dependency_source(
                re.value)
            if (isinstance(re.value, ast.Attribute)
                    and isinstance(container_dependency, ParameterSource)
                    and container_dependency.attributes):
                return InstanceMethod(
                    UnknownSource(
                        "item of parameter runtime attribute"),
                    method_name,
                )
            dependency = self._parameter_dependency_source(re)
            if isinstance(dependency, ParameterSource):
                if dependency.derived_operation == "slice":
                    return InstanceMethod(dependency, method_name)
                return InstanceMethod(
                    dependency.name,
                    method_name,
                    parameter_scope=dependency.scope,
                    parameter_name=dependency.name,
                )
            if isinstance(dependency, UnknownSource):
                return InstanceMethod(dependency, method_name)
            receiver_top = None
            if self._is_instance_field_expression(re.value):
                receiver_top = self._explicit_external_receiver_top(re.value)
            if receiver_top not in (None, "", "local", "python", "unknown"):
                return InstanceMethod(receiver_top, method_name)

        if isinstance(re, ast.Attribute):
            attribute_receiver_top = self._expr_receiver_top(re.value)
            attribute_owner = _match_attribute_result_owner(
                attribute_receiver_top, re.attr)
            if attribute_owner is not None:
                return InstanceMethod(attribute_owner, method_name)
            receiver_name = self._attribute_name(re)
            if (self._import_attribute_key(re) in
                    self._subscripted_import_attribute_receivers):
                return InstanceMethod(
                    UnknownSource(
                        "runtime attribute with subscript protocol"),
                    method_name,
                )
            if receiver_name:
                receiver_binding = self.current_scope().lookup(
                    receiver_name, skip_parent_classes=True)
                if (receiver_binding is not None
                        and receiver_binding.source == "python"):
                    return InstanceMethod("python", method_name)
            chain = self._attribute_chain_list(re)
            if chain:
                if chain[0] == "self" and self._class_stack:
                    cn = self._class_stack[-1]
                    attr_name = "self." + ".".join(chain[1:])
                    result = _resolve_on_class(cn, cn)
                    if isinstance(normalize_source(result), InstanceMethod):
                        if attr_name == "self.__dict__":
                            if _has_builtin_shape_method(
                                    "dict", method_name):
                                return InstanceMethod("python", method_name)
                            return InstanceMethod(
                                UnknownSource(
                                    "unsupported dict protocol"),
                                method_name,
                            )
                        attr_source = _lookup_instance_attr(attr_name)
                        attr_source = normalize_source(attr_source)
                        if isinstance(attr_source, ParameterSource):
                            receiver = ".".join(
                                (attr_source.name,) + attr_source.attributes)
                            return InstanceMethod(
                                receiver,
                                method_name,
                                parameter_scope=attr_source.scope,
                                parameter_name=attr_source.name)
                        if isinstance(attr_source, CallResult):
                            callee = attr_source.callee
                            if not isinstance(callee, str):
                                return InstanceMethod(
                                    attr_source, method_name)
                            callee_parts = callee.rsplit(".", 1)
                            is_local_method_result = (
                                len(callee_parts) == 2
                                and callee_parts[0] in self.class_methods
                                and callee_parts[1] in self.class_methods[
                                    callee_parts[0]]
                            )
                            if (callee in self.return_sources
                                    or is_local_method_result):
                                return InstanceMethod(
                                    attr_source, method_name)
                            if ('.' not in callee
                                    and (callee in self.symbols.direct
                                         or callee in self.import_from_symbols
                                         or any(
                                             alias == callee
                                             or alias.startswith(callee + ".")
                                             for alias in self.import_aliases))):
                                return InstanceMethod(callee, method_name)
                            if '.' in callee:
                                prefix = callee.split('.')[0]
                                prefix_is_origin = any(
                                    isinstance(v, str) and (v == prefix or v.startswith(prefix + "."))
                                    for v in self.symbols.direct.values())
                                if prefix_is_origin:
                                    return InstanceMethod(prefix, method_name)
                                if prefix in self.import_from_symbols:
                                    return InstanceMethod(prefix, method_name)
                                if prefix in self.symbols.direct:
                                    return InstanceMethod(prefix, method_name)
                                return InstanceMethod(callee, method_name)
                            wildcard_owner = (
                                self._unique_wildcard_import_owner())
                            if wildcard_owner is not None:
                                return InstanceMethod(
                                    wildcard_owner, method_name)
                        if isinstance(attr_source, str) and '.' not in attr_source and attr_source in self.symbols.direct:
                            return InstanceMethod(attr_source, method_name)
                        if isinstance(attr_source, str) and attr_source != "local":
                            return InstanceMethod(attr_source, method_name)
                    scope_name = (
                        self._caller_stack[-1].qualname
                        if self._caller_stack else "")
                    return InstanceMethod(
                        InstanceAttribute(cn, attr_name, scope_name),
                        method_name,
                    )
                root = chain[0]
                root_binding = self.current_scope().lookup(
                    root, skip_parent_classes=True)
                if (root_binding is not None
                        and root_binding.binding_kind == "import"
                        and root_binding.scope_kind != SCOPE_MODULE):
                    return InstanceMethod(root_binding.source, method_name)
                if root in self.import_from_symbols:
                    return InstanceMethod(root, method_name)
                root_src = self.symbols.direct.get(root)
                root_src = normalize_source(root_src)
                if isinstance(root_src, CallResult):
                    root_src = root_src.callee
                if root_src in self.import_from_symbols:
                    return InstanceMethod(root_src, method_name)
                field_source = None
                for scope_key in self._local_instance_field_scope_keys(root):
                    field_source = self._local_instance_field_sources.get(
                        (scope_key, root, ".".join(chain[1:])))
                    if field_source is not None:
                        break
                if field_source is not None:
                    field_top = self._structured_source_owner_top(field_source)
                    if field_top not in (
                            None, "", "local", "python", "unknown"):
                        return InstanceMethod(field_top, method_name)
                # P0: resolve instance attributes for non-self receivers
                # whose root traces to a local class instance.
                # e.g. client.backend.loads() where client=Client()
                # and self.backend=json  →  trace client.backend
                # through instance_attrs to find json, then
                # classify loads as json.loads.
                target_class = root_src
                if not target_class:
                    binding = self.current_scope().lookup(root)
                    if binding is not None:
                        target_class = normalize_source(binding.source)
                        if isinstance(target_class, CallResult):
                            target_class = target_class.callee
                        if target_class == "local":
                            parameter = _parameter_method(
                                ".".join(chain), root)
                            if parameter is not None:
                                return parameter
                if isinstance(target_class, str) and target_class in self.class_methods:
                    if len(chain) >= 2:
                        attr_name = "self." + ".".join(chain[1:])
                        attr_source = self.instance_attrs.get(
                            (target_class, attr_name))
                        attr_source = normalize_source(attr_source)
                        if isinstance(attr_source, CallResult):
                            callee = attr_source.callee
                            if isinstance(callee, str):
                                return InstanceMethod(callee, method_name)
                        if isinstance(attr_source, str) and attr_source not in ("local", "python", "unknown", ""):
                            return InstanceMethod(attr_source, method_name)
                    return _resolve_on_class(target_class, root)
        if isinstance(re, ast.Constant):
            # Literal constant method call: "str".format() → python.
            # The receiver is a literal Python builtin type, so the
            # method callable is always python.
            if isinstance(re.value, str):
                return InstanceMethod("str", method_name)
            if isinstance(re.value, bytes):
                return InstanceMethod("bytes", method_name)
        if (isinstance(re, ast.Subscript)
                and isinstance(re.value, ast.Call)):
            item_owner = self._resolve_call_result_item_owner(re.value)
            if item_owner is not None:
                return InstanceMethod(item_owner, method_name)
        if isinstance(re, ast.Compare):
            result = self._resolve_compare_result_top(re, method_name)
            if result is not None:
                return result
            # Owner could not be resolved — still collect the call
            # so it isn't silently dropped.  The cross-file resolver
            # will treat it conservatively.
            return InstanceMethod("__unresolved_compare__", method_name)
        return None

    ## Resolve the owner of a comparison-result method call.
    #
    #  For (np.diag(W) == np.zeros(...)).any(), both sides of the
    #  comparison are numpy expressions, so the result is a boolean
    #  ndarray and .any() belongs to numpy.
    #
    #  Only returns an owner when ALL operands resolve to the same
    #  library AND the method is in _COMPARE_RESULT_METHODS for that
    #  library.  Returns None otherwise — no fallback to the first
    #  operand (which would overclaim ownership for mixed libraries).
    #  @param compare_node The ast.Compare node.
    #  @param method_name The method being called on the result.
    #  @return InstanceMethod or None.
    def _resolve_compare_result_top(self, compare_node, method_name):
        # Collect the top library of every operand.
        operands = [compare_node.left] + list(compare_node.comparators)
        tops = []
        for op in operands:
            base = self.get_base(op)
            if isinstance(base, str):
                top = self.symbols.get_top(base)
                if top and top not in ("local", "python", "unknown", ""):
                    tops.append(top)
                    continue
            # Could not resolve this operand — conservative bail-out.
            return None

        if not tops:
            return None

        first = tops[0]
        # All operands must have the same owner.
        if any(t != first for t in tops[1:]):
            return None

        # Only allow methods known to exist on compare-result objects.
        allowed = _COMPARE_RESULT_METHODS.get(first)
        if allowed is None or method_name not in allowed:
            return None

        return InstanceMethod(first, method_name)

    ## Flatten an attribute chain (e.g. a.b.c) into a list ["a", "b", "c"].
    #  @param node The starting Attribute node.
    #  @return List of name parts from root to leaf, or None.
    def _attribute_chain_list(self, node):
        parts = []
        remain = node
        while isinstance(remain, ast.Attribute):
            parts.append(remain.attr)
            remain = remain.value
        if isinstance(remain, ast.Name):
            parts.append(remain.id)
            return list(reversed(parts))
        return None

    ## Reconstruct a dotted attribute name from an AST node.
    #  @param node The Attribute or Name node.
    #  @return Dotted name string (e.g. "os.path.join"), or None.
    def _attribute_name(self, node):
        chain = self._attribute_chain_list(node)
        if chain:
            return ".".join(chain)
        return None

    ## Check whether an expression is rooted at an imported module symbol.
    #
    #  This is deliberately syntactic.  A variable whose value came from an
    #  external call is not treated as the module that produced that value.
    #  @param node Receiver expression below an Attribute node.
    #  @return True only for an import alias or import-from root.
    def _is_import_backed_receiver_expression(self, node):
        if isinstance(node, ast.Name):
            root = node.id
        elif isinstance(node, ast.Attribute):
            chain = self._attribute_chain_list(node)
            root = chain[0] if chain else None
        else:
            root = None
        if not root:
            return False
        if (root not in self.import_aliases
                and root not in self.import_from_symbols):
            return False
        top = self._receiver_top(root)
        return top not in (None, "", "local", "python", "unknown")

    ## Resolve an explicitly evidenced external receiver expression.
    #
    #  Accepted roots are import symbols and instance fields already bound to
    #  an external source. A local variable is intentionally excluded even
    #  when its current source came from an external call, because that call's
    #  return object has no generic static owner contract.
    #  @param node Receiver expression.
    #  @return External owner top or None.
    def _explicit_external_receiver_top(self, node):
        if self._is_import_backed_receiver_expression(node):
            if isinstance(node, ast.Name):
                root = node.id
            else:
                chain = self._attribute_chain_list(node)
                root = chain[0] if chain else None
            return self._receiver_top(root) if root else None

        if isinstance(node, ast.Attribute) and self._class_stack:
            chain = self._attribute_chain_list(node) or []
            if chain and chain[0] == "self":
                class_name = self._class_stack[-1]
                for end in range(len(chain), 1, -1):
                    field_name = ".".join(chain[:end])
                    source = normalize_source(self.instance_attrs.get(
                        (class_name, field_name)))
                    owner = self._structured_source_owner_top(source)
                    if owner not in (None, "", "local", "python", "unknown"):
                        return owner

        if isinstance(node, ast.Subscript):
            return self._explicit_external_receiver_top(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if self._is_import_backed_receiver_expression(receiver):
                return None
            return self._explicit_external_receiver_top(receiver)
        if isinstance(node, ast.Attribute):
            return self._explicit_external_receiver_top(node.value)
        return None

    ## Check whether an expression is rooted at a local instance field.
    #  @param node Receiver expression.
    #  @return True for self.field and its subscripted forms.
    def _is_instance_field_expression(self, node):
        if isinstance(node, ast.Attribute):
            chain = self._attribute_chain_list(node)
            return bool(chain and chain[0] == "self")
        if isinstance(node, ast.Subscript):
            return self._is_instance_field_expression(node.value)
        return False

    ## Find the root receiver of a call expression.
    #
    #  Unwinds chained calls and attributes to find the base object.
    #  @param receiver_node The receiver AST node.
    #  @return Base symbol name, or None.
    ## Resolve an Attribute receiver through scope binding.
    #
    #  When the root of the attribute chain (e.g. "v" in "v.armW.mean")
    #  has a scope binding with a library source, propagate it instead
    #  of returning the raw dotted name.
    #  @param receiver_node The Attribute AST node.
    #  @param receiver_name The full dotted name (e.g. "v.armW").
    #  @return Resolved source or the original receiver_name.
    def _resolve_attribute_receiver_chain(self, receiver_node, receiver_name):
        if receiver_name in self.symbols.direct:
            return receiver_name
        chain = self._attribute_chain_list(receiver_node)
        if chain:
            root_src = self._lookup_name_source(chain[0])
            if root_src and root_src != chain[0]:
                return root_src
        return receiver_name

    def _resolve_call_receiver(self, receiver_node):
        if isinstance(receiver_node, ast.Name):
            return self._lookup_name_source(receiver_node.id)
        if isinstance(receiver_node, ast.Attribute):
            receiver_name = self._attribute_name(receiver_node)
            if receiver_name is not None:
                return self._resolve_attribute_receiver_chain(
                    receiver_node, receiver_name)
            return self._resolve_call_receiver(receiver_node.value)
        if isinstance(receiver_node, ast.Call):
            inner_receiver = self.get_base(receiver_node, call_lookup=True)
            if inner_receiver is not None:
                return inner_receiver
            return self.get_base(receiver_node.func, call_lookup=False)
        if isinstance(receiver_node, ast.BinOp):
            left = self.get_base(receiver_node.left, call_lookup=True)
            if left is not None:
                return left
            return self.get_base(receiver_node.right, call_lookup=True)
        if isinstance(receiver_node, ast.Subscript):
            return self._resolve_call_receiver(receiver_node.value)
        return None

    ## --- Decorator binding ---

    ## Record decorator evidence without overwriting the target's primary binding.
    #
    #  Each decorator expression is traced and recorded as a separate
    #  provenance record (kind="decorated_by"), while the decorated
    #  function/class keeps its "local" primary identity.
    #  @param target_name Name of the decorated function/class.
    #  @param decorator_nodes List of decorator AST nodes.
    def _bind_decorated_target(self, target_name, decorator_nodes):
        if not decorator_nodes:
            return
        for deco in reversed(decorator_nodes):
            deco_source = self.trace_source(deco)
            if not deco_source or (isinstance(deco_source, str) and _is_builtin(deco_source)):
                continue
            if deco_source == "local" and isinstance(deco, ast.Name):
                fn = deco.id
                rs = self.return_sources.get(fn)
                if rs is not None and not (isinstance(rs, str) and rs == "local"):
                    deco_source = rs
                else:
                    deco_source = fn
            self._add_symbol_ref(
                target_name, deco_source, "decorated_by", deco)

    ## --- Assignment helpers ---

    ## Record a project-local callable assigned onto an imported class.
    #
    #  The fact identifies a possible monkey patch. It does not prove that an
    #  arbitrary receiver from the same library has the patched runtime class;
    #  cross-file classification therefore uses it only to remove false
    #  certainty from a library owner.
    #  @param node Assignment node.
    def _record_external_method_override(self, node):
        local_callable = False
        if isinstance(node.value, ast.Lambda):
            local_callable = True
        elif isinstance(node.value, ast.Name):
            binding = self.current_scope().lookup(
                node.value.id, skip_parent_classes=True)
            local_callable = (
                binding is not None
                and (
                    bool(binding.callable_key)
                    or binding.source == "local"
                )
            )
        if not local_callable:
            return

        wildcard_tops = {
            module.split(".")[0] for module in self.wildcard_modules
            if isinstance(module, str) and module
        }
        scope_name = (
            self.current_scope().name
            if self.current_scope().kind != SCOPE_MODULE else "")
        for target in node.targets:
            if (not isinstance(target, ast.Attribute)
                    or not isinstance(target.value, ast.Name)):
                continue
            class_symbol = target.value.id
            qualified = self.import_from_symbols.get(class_symbol)
            if qualified:
                owner = qualified.split(".")[0]
            else:
                direct = normalize_source(
                    self.symbols.direct.get(class_symbol))
                if (isinstance(direct, str)
                        and direct not in (
                            "", "local", "python", "unknown")):
                    owner = direct.split(".")[0]
                elif len(wildcard_tops) == 1:
                    owner = next(iter(wildcard_tops))
                else:
                    continue
            key = (owner, target.attr)
            self.external_method_overrides.setdefault(key, []).append(
                (scope_name, node.lineno, class_symbol))

    ## Bind assignment targets to a source value.
    #
    #  Handles simple names, self.attr, and tuple/list unpacking.
    #  @param target The assignment target AST node.
    #  @param source The source symbol or structured tuple.
    def _target_to_source(self, target, source, kind="variable",
                          container_kind="", container_item_kind="",
                          container_item_fields=None):
        if not source:
            return
        if isinstance(target, ast.Name):
            self._bind_target_name(
                target.id, source, target, kind,
                container_kind=container_kind,
                container_item_kind=container_item_kind,
                container_item_fields=container_item_fields)
            return
        if isinstance(target, ast.Attribute):
            name = self._attribute_name(target)
            attr_name = name if name and name.startswith("self.") else (
                self._instance_attribute_target_name(name))
            if attr_name:
                self._bind_target_name(attr_name, source, target, "attribute")
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._target_to_source(
                    elt, source, kind,
                    container_kind=container_kind,
                    container_item_kind=container_item_kind,
                    container_item_fields=container_item_fields)

    ## Trace the source of a for-loop iterator.
    #  @param iter_node The iterator AST node.
    #  @return Source symbol, structured tuple, or None.
    def _iter_source(self, iter_node):
        if isinstance(iter_node, ast.Name):
            container_name = iter_node.id
            binding = self.current_scope().lookup(
                container_name, skip_parent_classes=True)
            item_source = (
                self._iterated_append_sources.get(
                    self._binding_key(binding))
                if binding is not None else None)
            if item_source is not None:
                return item_source
            item_source = self.homogeneous_container_items.get(
                container_name)
            if item_source is not None:
                if binding is None or binding.scope_kind == SCOPE_MODULE:
                    return item_source
            has_items = False
            for k in self.container_items.keys():
                if k[0] == container_name:
                    has_items = True
                    break
            has_set = container_name in self.container_set_sources
            if ((has_items or has_set) and binding is not None
                    and binding.container_kind in ("dict", "list", "tuple", "set")):
                return ContainerIter(container_name)
        parameter_source = self._parameter_dependency_source(iter_node)
        if parameter_source is not None:
            return ContainerIter(parameter_source)
        source = self.trace_source(iter_node)
        source_norm = normalize_source(source)
        if (isinstance(source_norm, CallResult)
                and isinstance(source_norm.callee, str)
                and source_norm.callee in self.return_element_sources):
            return ContainerIter(source_norm)
        if (isinstance(source_norm, CallResult)
                and isinstance(
                    normalize_source(source_norm.callee), ContainerIter)
                and source_norm.result_source is None):
            return UnknownSource("unresolved iterator element")
        if isinstance(normalize_source(source), ParameterSource):
            return ContainerIter(source)
        if source:
            return source
        return self.get_base(iter_node)

    ## --- Base extraction ---

    ## Extract the root/base name from an expression node.
    #
    #  For simple names returns the name. For attributes returns the chain root.
    #  For calls with call_lookup=True, resolves the call receiver.
    #  @param node The AST expression node.
    #  @param call_lookup If True, resolve call receivers instead of just func base.
    #  @return Root symbol name, or None.
    def get_base(self, node, call_lookup=False):
        if isinstance(node, ast.Name):
            return self._lookup_name_source(node.id)
        elif isinstance(node, ast.Attribute):
            chain = self._attribute_chain_list(node)
            if chain:
                name = '.'.join(chain)
                if name in self.symbols.direct:
                    return name
                if chain[0] == "self" and self._class_stack:
                    cn = self._class_stack[-1]
                    attr_source = self.instance_attrs.get((cn, name))
                    if attr_source is not None:
                        if isinstance(attr_source, str):
                            if attr_source in self.symbols.direct or '.' in attr_source:
                                return attr_source
                        else:
                            return attr_source
                root = chain[0]
                return self._lookup_name_source(root)
            return self.get_base(node.value, call_lookup=call_lookup)
        elif isinstance(node, ast.Call):
            if self._is_partial_call(node) and node.args:
                return self.get_base(node.args[0], call_lookup=call_lookup)
            if call_lookup:
                func = node.func
                if isinstance(func, ast.Attribute):
                    return self._resolve_call_receiver(func.value)
                if isinstance(func, ast.Call):
                    return self._resolve_call_receiver(func)
                if isinstance(func, ast.Name):
                    return self._lookup_name_source(func.id)
                return None
            return self.get_base(node.func, call_lookup=False)
        elif isinstance(node, ast.BinOp):
            left = self.get_base(node.left, call_lookup=call_lookup)
            if left is not None:
                return left
            return self.get_base(node.right, call_lookup=call_lookup)
        elif isinstance(node, ast.Lambda):
            return self.get_base(node.body, call_lookup=call_lookup)
        elif isinstance(node, ast.Subscript):
            return self.get_base(node.value, call_lookup=call_lookup)
        return None

    ## --- Visit handlers ---

    ## Visit an Assign node and record symbol bindings.
    #
    #  Handles dict/list/tuple/set container tracking, and traces the
    #  right-hand side to bind target symbols.
    #  @param node The Assign AST node.
    def visit_Assign(self, node):
        mapping_value = self._mapping_facts.value(node.value)
        assignment_container_kind, assignment_item_kind = (
            self._expression_container_shape(node.value))
        assignment_item_fields = self._expression_container_item_fields(
            node.value)
        assignment_container_kind = assignment_container_kind or None
        assignment_item_kind = assignment_item_kind or None

        # Homogeneous comprehension evidence is flow-sensitive at module
        # scope. Any real rebind invalidates the previous element source.
        if self.current_scope().kind == SCOPE_MODULE:
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                is_self_assignment = (
                    isinstance(node.value, ast.Name)
                    and node.value.id == target.id
                )
                if not is_self_assignment:
                    self.homogeneous_container_items.pop(
                        target.id, None)
                    self.homogeneous_container_tuple_items.pop(
                        target.id, None)
                    binding = self.current_scope().lookup(
                        target.id, skip_parent_classes=True)
                    if binding is not None:
                        binding_key = self._binding_key(binding)
                        self._iterated_append_tuple_sources.pop(
                            binding_key, None)
                        self._iterated_append_tuple_conflicts.discard(
                            binding_key)

        ## Track literal assignments for static key resolution (PR7).
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, int)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._literal_values[target.id] = node.value.value

        if isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    container_name = target.id
                    self.container_kinds[container_name] = "dict"
                    value_sources = []
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        if isinstance(key_node, ast.Constant):
                            key_value = key_node.value
                            if isinstance(value_node, ast.Constant):
                                value_source = "python"
                            else:
                                value_source = self._value_source(value_node)
                            if value_source:
                                self.container_items[(container_name, key_value)] = value_source
                                value_sources.append(normalize_source(value_source))
                    owners = {
                        self._structured_source_owner_top(source)
                        for source in value_sources
                    }
                    owners.discard(None)
                    if (value_sources
                            and len(value_sources) == len(node.value.values)
                            and len(owners) == 1):
                        self.homogeneous_container_value_sources[
                            container_name] = make_source_set(
                                value_sources, origin="dict_values")
                    else:
                        self.homogeneous_container_value_sources.pop(
                            container_name, None)

        if isinstance(node.value, (ast.List, ast.Tuple)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    container_name = target.id
                    n = len(node.value.elts)
                    self.container_lengths[container_name] = n
                    self.container_kinds[container_name] = (
                        "list" if isinstance(node.value, ast.List) else "tuple")
                    for i, elt in enumerate(node.value.elts):
                        value_source = _builtin_value_source(
                            elt, self.get_base)
                        if value_source:
                            self.container_items[(container_name, i)] = value_source

        if isinstance(node.value, ast.Set):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    container_name = target.id
                    self.container_kinds[container_name] = "set"
                    bases = set()
                    for elt in node.value.elts:
                        base = _builtin_value_source(elt, self.get_base)
                        if base:
                            bases.add(base)
                    if bases:
                        self.container_set_sources[container_name] = bases

        # 1.0.5 P1: track container kinds for comprehensions and constructors.
        if isinstance(node.value, ast.ListComp):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.container_kinds[target.id] = "list"
                    if self.current_scope().kind == SCOPE_MODULE:
                        tuple_sources = self._expression_tuple_item_sources(
                            node.value)
                        if tuple_sources is not None:
                            self.homogeneous_container_tuple_items[
                                target.id] = tuple_sources
                        else:
                            self.homogeneous_container_tuple_items.pop(
                                target.id, None)
                    else:
                        self.homogeneous_container_tuple_items.pop(
                            target.id, None)
                    if self.current_scope().kind == SCOPE_MODULE:
                        item_source = self.trace_source(node.value.elt)
                        if item_source is not None:
                            self.homogeneous_container_items[
                                target.id] = item_source
        if isinstance(node.value, ast.SetComp):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.container_kinds[target.id] = "set"
        if isinstance(node.value, ast.DictComp):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.container_kinds[target.id] = "dict"
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.container_kinds[target.id] = "str"
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            func_id = node.value.func.id
            if func_id in ("list", "dict", "set", "tuple", "str"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.container_kinds[target.id] = func_id
            # 1.0.5 P1+: track defaultdict(list) / defaultdict(dict) etc.
            # so d[k].append(v) can be classified from the item kind.
            if func_id == "defaultdict" and len(node.value.args) >= 1:
                factory = node.value.args[0]
                if isinstance(factory, ast.Name):
                    factory_name = factory.id
                    if factory_name in ("list", "dict", "set", "tuple", "str"):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.container_item_kinds[target.id] = factory_name

        # Keep compatibility maps flow-sensitive. Lexical Binding metadata is
        # authoritative, while these maps support older module-level paths.
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if assignment_container_kind:
                self.container_kinds[
                    target.id] = assignment_container_kind
            else:
                self.container_kinds.pop(target.id, None)
            if assignment_item_kind:
                self.container_item_kinds[
                    target.id] = assignment_item_kind
            else:
                self.container_item_kinds.pop(target.id, None)

        ## Collect assignment target names and delegate to shared pipeline.
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)
        field_targets = []
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                name = self._attribute_name(target)
                if name and name.startswith("self."):
                    field_targets.append(name)
        right = self._visit_assignment(node, targets, field_targets)
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                self._invalidate_attribute_tuple_item_sources(target)
        self._record_local_constructor_fields(node, targets)
        self._record_external_method_override(node)
        callable_keys = {}
        if isinstance(node.value, ast.Lambda):
            lambda_result = right or UnknownSource("lambda result")
            lambda_key = self._local_lambda_key(node.value)
            for name in targets:
                callable_keys[name] = lambda_key
            self.return_sources[lambda_key] = lambda_result
            # The lambda expression defines a project-local callable. Its
            # body source describes only the object returned by that call.
            right = "local"
        if right:
            right_norm = normalize_source(right)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if (
                        isinstance(right_norm, InstanceMethod)
                        and right_norm.receiver == target.id
                    ):
                        continue
                    if (
                        isinstance(right_norm, CallResult)
                        and (right_norm.callee == target.id or (isinstance(right_norm.callee, str) and right_norm.callee.startswith(target.id + ".")))
                    ):
                        continue
                    ## skip self-assign: df = df[...] where right resolves to "df"
                    if isinstance(right, str) and right == target.id:
                        continue
                    self._bind_target_name(
                        target.id, right, target,
                        container_kind=assignment_container_kind or "",
                        container_item_kind=assignment_item_kind or "",
                        callable_key=callable_keys.get(target.id, ""),
                        container_item_fields=assignment_item_fields)
                elif isinstance(target, ast.Attribute):
                    name = self._attribute_name(target)
                    attr_name = name if name and name.startswith("self.") else (
                        self._instance_attribute_target_name(name))
                    if attr_name:
                        if isinstance(right_norm, InstanceMethod):
                            preserve_imported_result = (
                                isinstance(node.value, ast.Call)
                                and isinstance(node.value.func, ast.Attribute)
                                and self._is_import_backed_receiver_expression(
                                    node.value.func.value)
                                and self._resolve_func_top(
                                    node.value.func)[0]
                                == self._structured_source_owner_top(
                                    right_norm.receiver))
                            if not preserve_imported_result:
                                continue
                        attr_source = right
                        if (isinstance(node.value, ast.Name)
                                and right == "local"):
                            attr_source = node.value.id
                        self._bind_target_name(
                            attr_name, attr_source, target,
                            container_kind=assignment_container_kind or "",
                            container_item_kind=assignment_item_kind or "",
                            container_item_fields=assignment_item_fields)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    unpacked_owner = None
                    if isinstance(node.value, ast.Call):
                        func_top, func_name = self._resolve_func_top(
                            node.value.func)
                        unpacked_owner = _match_result_item_owner(
                            func_top, func_name)
                    for index, elt in enumerate(target.elts):
                        if isinstance(elt, ast.Name):
                            if (isinstance(node.value, (ast.Tuple, ast.List))
                                    and index < len(node.value.elts)):
                                dependency = self._parameter_dependency_source(
                                    node.value.elts[index])
                                if dependency is not None:
                                    self._bind_target_name(
                                        elt.id, dependency, elt)
                                    continue
                            if unpacked_owner is not None:
                                self._bind_target_name(
                                    elt.id, unpacked_owner, elt)
                                continue
                            if (isinstance(node.value, ast.Call)
                                    and isinstance(right_norm, CallResult)):
                                # Keep the selected position when a local
                                # tuple-return call is forwarded as an
                                # argument before project summaries exist.
                                selected_source = ContainerItem(
                                    right_norm, index)
                                self._bind_target_name(
                                    elt.id, right, elt)
                                self._assigned_call_sources[
                                    (id(self.current_scope()), elt.id)
                                ] = selected_source
                                continue
                            # Preserve positional provenance when unpacking a
                            # named value.  Flattening every element to the
                            # traced top (often "local" for a parameter)
                            # allows a module-level symbol with the same name
                            # to leak back in during cross-file resolution.
                            if isinstance(node.value, ast.Name):
                                dependency = self._parameter_dependency_source(
                                    node.value)
                                self._bind_target_name(
                                    elt.id,
                                    ContainerItem(
                                        dependency or node.value.id, index),
                                    elt)
                                continue
                            if isinstance(right_norm, InstanceMethod):
                                if isinstance(right_norm.receiver, str):
                                    # 1.0.5 P1: consult result-owner map
                                    # before binding.  linalg.svd(arr)
                                    # returns numpy arrays even though
                                    # linalg is scipy.
                                    rcvr_top = self.symbols.get_top(
                                        right_norm.receiver)
                                    if rcvr_top:
                                        ret = _match_result_item_owner(
                                            rcvr_top, right_norm.method)
                                        if ret:
                                            self._bind_target_name(
                                                elt.id, ret, elt)
                                            continue
                                    self._bind_target_name(
                                        elt.id, right_norm.receiver, elt)
                                continue
                            self._bind_target_name(elt.id, right, elt)
        else:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._bind_target_name(
                        target.id, 'local', target,
                        container_kind=assignment_container_kind or "",
                        container_item_kind=assignment_item_kind or "",
                        callable_key=callable_keys.get(target.id, ""),
                        container_item_fields=assignment_item_fields)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self._bind_target_name(elt.id, 'local', elt)
                elif isinstance(target, ast.Attribute):
                    name = self._attribute_name(target)
                    if name and name.startswith("self."):
                        self._bind_target_name(
                            name, 'local', target,
                            container_kind=assignment_container_kind or "",
                            container_item_kind=assignment_item_kind or "",
                            container_item_fields=assignment_item_fields)
        subscript_value_kind = assignment_container_kind
        if (subscript_value_kind is None
                and isinstance(node.value, ast.Name)):
            subscript_value_kind = self._lookup_container_kind(
                node.value.id)
        for target in node.targets:
            self._record_subscript_item_kind(
                target, subscript_value_kind)
            self._bind_mapping_value(target, mapping_value)
        self._record_iterable_binding_source(node)
        # 1.0.5 P0: generic_visit already called above, before target binding.
        self._collect_argparse_assignment(node)

    ## Resolve a receiver name to its top library using lexical scope.
    def _receiver_top(self, name):
        binding = self.current_scope().lookup(
            name, skip_parent_classes=True)
        if binding is not None:
            src = normalize_source(binding.source)
            if isinstance(src, str):
                if src in ("local", "python", "unknown", ""):
                    return src or None
                source_top = self.symbols.get_top(src)
                return source_top or src
            if isinstance(src, InstanceMethod):
                receiver = normalize_source(src.receiver)
                if isinstance(receiver, str):
                    receiver_top = self.symbols.get_top(receiver)
                    return receiver_top or receiver
                return None
            if isinstance(src, CallResult):
                result_owner = normalize_source(src.result_source)
                if isinstance(result_owner, str):
                    owner_top = self.symbols.get_top(result_owner)
                    return owner_top or result_owner
                if isinstance(result_owner, UnknownSource):
                    return "unknown"
                if not isinstance(src.callee, str):
                    return None
                callee_top = self.symbols.get_top(src.callee)
                root = src.callee.split(".", 1)[0]
                root_binding = self.current_scope().lookup(
                    root, skip_parent_classes=True)
                if ("." in src.callee and root_binding is not None
                        and root_binding.binding_kind == "import"
                        and isinstance(root_binding.source, str)):
                    callee_top = root_binding.source.split(".", 1)[0]
                if callee_top and callee_top not in ("local", name):
                    return callee_top
                # Follow return_sources through local functions.
                rs = self.return_sources.get(src.callee)
                if rs is not None:
                    rs = normalize_source(rs)
                    sources = rs.sources if isinstance(rs, SourceSet) else [rs]
                    for source in sources:
                        source = normalize_source(source)
                        if (isinstance(source, CallResult)
                                and isinstance(source.callee, str)):
                            callee_top = self.symbols.get_top(source.callee)
                            if (callee_top
                                    and callee_top not in ("local", name)):
                                return callee_top
                return callee_top
            # A lexical binding is authoritative. Do not consult a same-name
            # module binding when its structured source is unresolved here.
            return None
        top = self.symbols.get_top(name)
        return top

    ## Resolve the ownership top of an expression node.
    #  Recursively handles ast.Call arguments so that
    #  np.log(price.dropna()).diff() preserves the inner call's
    #  receiver owner (pandas).
    #  Respects conversion boundaries: data.to_numpy() returns
    #  numpy even though the receiver is pandas.
    def _expr_receiver_top(self, expr):
        if isinstance(expr, ast.Name):
            return self._receiver_top(expr.id)
        if isinstance(expr, ast.Constant):
            return "python"
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return "python"
        if isinstance(expr, ast.UnaryOp):
            return self._expr_receiver_top(expr.operand)
        if isinstance(expr, ast.BinOp):
            left_top = self._expr_receiver_top(expr.left)
            right_top = self._expr_receiver_top(expr.right)
            if left_top == right_top:
                return left_top
            external = [
                top for top in (left_top, right_top)
                if top not in (None, "", "local", "python", "unknown")]
            other = [
                top for top in (left_top, right_top)
                if top not in external]
            if (len(set(external)) == 1
                    and all(top == "python" for top in other)):
                return external[0]
            return None
        if isinstance(expr, ast.Subscript):
            return self._expr_receiver_top(expr.value)
        if isinstance(expr, ast.Attribute):
            receiver = expr.value
            name = self._attribute_name(expr)
            if (name and name.startswith("self.")
                    and self._class_stack):
                field_source = normalize_source(self.instance_attrs.get(
                    (self._class_stack[-1], name)))
                field_top = self._structured_source_owner_top(field_source)
                if field_top not in (None, "", "local", "python", "unknown"):
                    return field_top
            if isinstance(receiver, ast.Name):
                receiver_top = self._receiver_top(receiver.id)
                if receiver_top:
                    return _CONVERSION_ATTRIBUTE_TARGETS.get(
                        (receiver_top, expr.attr))
            return None
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            receiver_top = self._expr_receiver_top(expr.func.value)
            if (receiver_top
                    and receiver_top not in (
                        "local", "python", "unknown", "")):
                # Check conversion boundary first:
                # data.to_numpy() return is numpy, not pandas.
                conv = _CONVERSION_METHOD_TARGETS.get(
                    (receiver_top, expr.func.attr))
                if conv:
                    return conv
                return receiver_top
        return None

    ## Identify one imported attribute path by its lexical root binding.
    #  @param node Attribute expression.
    #  @return Binding key and attribute path, or None for non-import roots.
    def _import_attribute_key(self, node):
        if not isinstance(node, ast.Attribute):
            return None
        chain = self._attribute_chain_list(node)
        if not chain:
            return None
        binding = self.current_scope().lookup(
            chain[0], skip_parent_classes=True)
        if binding is None or binding.binding_kind != "import":
            return None
        return (self._binding_key(binding), tuple(chain[1:]))

    ## Record runtime attributes of imported objects used via subscripting.
    #
    #  A subscript proves only that the attribute supports the subscription
    #  protocol. It does not prove that the attribute value is owned by the
    #  library that owns the outer object. Later method calls on the same
    #  attribute therefore remain unknown unless independent shape evidence
    #  exists.
    #  @param node Subscript expression.
    def visit_Subscript(self, node):
        key = self._import_attribute_key(node.value)
        if key is not None:
            self._subscripted_import_attribute_receivers.add(key)
        self.generic_visit(node)

    ## Check whether a return expression is fully bounded by project sources.
    #  @param source Structured return-expression source.
    #  @return True when exact call-edge substitution is safe.
    def _is_bounded_return_expression(self, source):
        source = normalize_source(source)
        if isinstance(source, (ParameterSource, InstanceAttribute,
                               PythonShape)):
            return True
        if isinstance(source, str):
            return source == "python"
        if isinstance(source, DerivedResult):
            return (
                source.kind == "expression"
                and bool(source.sources)
                and all(self._is_bounded_return_expression(item)
                        for item in source.sources)
            )
        if isinstance(source, CallResult):
            return (
                source.result_source is not None
                and self._is_bounded_return_expression(
                    source.result_source)
            )
        return False

    ## Check whether a structured source still depends on a parameter.
    #
    #  Parameter expressions must be resolved from project call edges.  A
    #  same-scope expression owner must not overwrite that richer evidence.
    #  @param source Source value to inspect.
    #  @return True when any nested source is parameter-backed.
    def _source_contains_parameter(self, source):
        source = normalize_source(source)
        if isinstance(source, ParameterSource):
            return True
        if isinstance(source, (DerivedResult, SourceSet)):
            return any(self._source_contains_parameter(item)
                       for item in source.sources)
        if isinstance(source, CallResult):
            return self._source_contains_parameter(source.result_source)
        if isinstance(source, InstanceMethod):
            return self._source_contains_parameter(source.receiver)
        return False

    ## Collect immediate ownership evidence from an operator expression.
    #
    #  The caller uses these candidates only to distinguish strict
    #  same-owner convergence from conflicting import-backed operands.
    #  @param expr BinOp or UnaryOp expression.
    #  @return List of operand owner strings.
    def _operator_operand_tops(self, expr):
        if isinstance(expr, ast.BinOp):
            return [
                self._expr_receiver_top(expr.left),
                self._expr_receiver_top(expr.right),
            ]
        if isinstance(expr, ast.UnaryOp):
            return [self._expr_receiver_top(expr.operand)]
        return []

    ## Resolve (top_library, function_name) for a function expression.
    #  Handles both bare names (cdist) and dotted names (np.log).
    def _resolve_func_top(self, func_node):
        if isinstance(func_node, ast.Name):
            name = func_node.id
            top = self._receiver_top(name)
            imported_name = self.import_from_symbols.get(name, "")
            func_name = (
                imported_name.rsplit(".", 1)[-1]
                if imported_name else name)
            if (top and top not in ("local", "python", "unknown", "")
                    and (top != name
                         or name in self.import_aliases
                         or name in self.import_from_symbols)):
                return (top, func_name)
        if isinstance(func_node, ast.Attribute):
            full_name = self._attribute_name(func_node)
            prefix = ""
            if full_name:
                imported_prefixes = sorted(
                    (
                        name for name in self.import_aliases
                        if (full_name == name
                            or full_name.startswith(name + "."))
                    ),
                    key=len,
                    reverse=True,
                )
                if imported_prefixes:
                    prefix = imported_prefixes[0]
                else:
                    prefix = full_name.split(".", 1)[0]
            prefix_top = self._receiver_top(prefix)
            if (prefix_top
                    and prefix_top not in ("local", "python", "unknown", "")
                    and (prefix_top != prefix
                         or prefix in self.import_aliases
                         or prefix in self.import_from_symbols)):
                return (prefix_top, func_node.attr)
        return (None, None)

    ## Resolve a true-branch receiver-owner guard.
    #
    #  @param test_node Conditional expression.
    #  @return (receiver_name, owner) or None.
    def _resolve_receiver_owner_guard(self, test_node):
        if (not isinstance(test_node, ast.Call)
                or not test_node.args
                or not isinstance(test_node.args[0], ast.Name)):
            return None
        if (_is_unshadowed_builtin_call(self, test_node)
                and isinstance(test_node.func, ast.Name)
                and test_node.func.id == "isinstance"
                and len(test_node.args) == 2
                and isinstance(test_node.args[1], ast.Name)
                and test_node.args[1].id in (
                    "list", "dict", "set", "tuple", "str")):
            return (
                test_node.args[0].id,
                PythonShape(test_node.args[1].id),
            )
        if len(test_node.args) != 1:
            return None
        func_top, func_name = self._resolve_func_top(test_node.func)
        for (lib_prefix, name), contract in (
                _TYPE_GUARD_OWNER_CONTRACTS.items()):
            if (name == func_name
                    and func_top is not None
                    and (func_top == lib_prefix
                         or func_top.startswith(lib_prefix + "."))):
                return (test_node.args[0].id, contract[0])
        return None

    ## Visit nodes under an optional receiver-owner guard.
    #
    #  @param nodes Iterable of AST nodes.
    #  @param guard Optional (receiver_name, owner) pair.
    #  @param test Optional branch condition for finite-name narrowing.
    def _visit_guarded_nodes(self, nodes, guard, test=None):
        finite = self._finite_name_guard(test, nodes)
        if finite is not None:
            self._finite_name_guards.append(finite)
        if guard is not None:
            self._receiver_owner_guards.append({guard[0]: guard[1]})
        try:
            for child in nodes:
                self.visit(child)
        finally:
            if guard is not None:
                self._receiver_owner_guards.pop()
            if finite is not None:
                self._finite_name_guards.pop()

    ## Read a finite string-name guard without evaluating project code.
    #  @param test Branch condition.
    #  @param nodes Guarded statements.
    #  @return Expression key and allowed strings, or None.
    def _finite_name_guard(self, test, nodes):
        if (not isinstance(test, ast.Compare) or len(test.ops) != 1
                or not isinstance(test.ops[0], ast.In)
                or not isinstance(test.left, (ast.Name, ast.Attribute))
                or not isinstance(test.comparators[0], (ast.List, ast.Tuple, ast.Set))):
            return None
        choices = test.comparators[0].elts
        if (not choices or len(choices) > 32 or any(
                not isinstance(item, ast.Constant) or not isinstance(item.value, str)
                for item in choices)):
            return None
        root = test.left
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            return None
        if isinstance(test.left, ast.Name):
            shape = self._expression_python_shape(test.left)
            if shape is None or shape.kind != "str":
                return None
        if (isinstance(test.left, ast.Attribute)
                and self._expression_python_shape(test.left) is None):
            field = self.current_scope().lookup(
                self._attribute_name(test.left), skip_parent_classes=True)
            if field is None or field.binding_kind != 'attribute' or field.source != 'python':
                return None
        mutated_roots = set()

        def escaped_roots(value):
            if isinstance(value, ast.Name):
                return {value.id}
            if isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                return set().union(*(escaped_roots(child)
                                     for child in ast.iter_child_nodes(value)))
            return set()

        for statement in nodes:
            for child in ast.walk(statement):
                if (isinstance(child, ast.Name) and child.id == root.id
                        and isinstance(child.ctx, (ast.Store, ast.Del))):
                    return None
                if isinstance(child, ast.Attribute) and isinstance(child.ctx, (ast.Store, ast.Del)):
                    base = child.value
                    while isinstance(base, ast.Attribute):
                        base = base.value
                    if isinstance(base, ast.Name):
                        mutated_roots.add(base.id)
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    mutated_roots.update(escaped_roots(child.value))
                if isinstance(child, ast.Call):
                    for arg in list(child.args) + [kw.value for kw in child.keywords]:
                        mutated_roots.update(escaped_roots(arg))
                    if isinstance(child.func, ast.Attribute):
                        base = child.func.value
                        while isinstance(base, ast.Attribute):
                            base = base.value
                        if isinstance(base, ast.Name) and base.id == root.id:
                            return None
                if isinstance(child, ast.Call) and any(
                        isinstance(arg, ast.Name) and arg.id == root.id
                        for arg in list(child.args) + [kw.value for kw in child.keywords]
                        ):
                    if not (_is_unshadowed_builtin_call(self, child)
                            and child.func.id == 'eval'):
                        return None
        if root.id in mutated_roots:
            return None
        return (ast.dump(test.left, include_attributes=False),
                tuple(item.value for item in choices), id(self.current_scope()),
                mutated_roots)

    ## Resolve only finite qualified-name eval expressions as value identities.
    #  @param node Call expression.
    #  @return SourceSet of possible names, or None for open evaluation.
    def _finite_eval_names(self, node):
        if (not self._finite_name_guards or len(node.args) != 1 or node.keywords
                or not _is_unshadowed_builtin_call(self, node)
                or node.func.id != 'eval' or not isinstance(node.args[0], ast.JoinedStr)):
            return None
        candidates = ['']
        mutated_roots = set()
        for part in node.args[0].values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                values = (part.value,)
            elif (isinstance(part, ast.FormattedValue) and part.conversion == -1
                  and part.format_spec is None):
                key = ast.dump(part.value, include_attributes=False)
                guard = next((guard for guard in reversed(self._finite_name_guards)
                              if guard[0] == key and guard[2] == id(self.current_scope())), None)
                values = guard[1] if guard is not None else None
                if values is None:
                    return None
                mutated_roots.update(guard[3])
            else:
                return None
            candidates = [prefix + value for prefix in candidates for value in values]
            if len(candidates) > 32:
                return None
        sources = []
        for candidate in candidates:
            try:
                expression = ast.parse(candidate, mode='eval').body
            except SyntaxError:
                return None
            if not isinstance(expression, ast.Attribute):
                return None
            parts = self._attribute_chain_list(expression)
            if not parts or parts[0] not in self.import_aliases:
                return None
            if parts[0] in mutated_roots:
                return None
            if not self._finite_namespace_is_stable(parts[0]):
                return None
            binding = self.current_scope().lookup(parts[0], skip_parent_classes=True)
            if binding is None or binding.binding_kind != 'import':
                return None
            sources.append('.'.join(parts))
        return make_source_set(sources, origin='finite_name_selection')

    ## Reject namespace writes or escapes before narrowing an eval result.
    #  @param name Imported module binding name.
    #  @return True only for read-only attribute access in this module.
    def _finite_namespace_is_stable(self, name):
        if name in self._finite_namespace_stability:
            return self._finite_namespace_stability[name]
        tree = self._module_tree
        if tree is None:
            return False
        parents = {id(child): node for node in ast.walk(tree)
                   for child in ast.iter_child_nodes(node)}
        stable = True
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id != name:
                continue
            parent = parents.get(id(node))
            if (not isinstance(node.ctx, ast.Load)
                    or not isinstance(parent, ast.Attribute)
                    or parent.value is not node):
                stable = False
                break
            while isinstance(parent, ast.Attribute):
                if not isinstance(parent.ctx, ast.Load):
                    stable = False
                    break
                node = parent
                parent = parents.get(id(node))
            if isinstance(parent, ast.Call) and parent.func is node:
                stable = False
            if not stable:
                break
        self._finite_namespace_stability[name] = stable
        return stable

    ## Resolve a verified owner for an item selected from a call result.
    #
    #  Direct import calls use _resolve_func_top().  Receiver calls additionally
    #  use _resolve_methods(), which can recover owners stored on instance
    #  attributes such as self.model = GPRegression(...).
    #  @param call_node The call whose result is indexed or destructured.
    #  @return Verified item owner string, or None.
    def _resolve_call_result_item_owner(self, call_node):
        func_top, func_name = self._resolve_func_top(call_node.func)
        owner = _match_result_item_owner(func_top, func_name)
        if owner is not None:
            return owner
        method_source = normalize_source(self._resolve_methods(call_node))
        if isinstance(method_source, InstanceMethod):
            receiver = normalize_source(method_source.receiver)
            if isinstance(receiver, str):
                receiver_top = self.symbols.get_top(receiver) or receiver
                if receiver_top not in (
                        None, "", "local", "python", "unknown"):
                    return _match_result_item_owner(
                        receiver_top, method_source.method)
        if (not isinstance(call_node.func, ast.Attribute)
                or not isinstance(call_node.func.value, ast.Attribute)
                or not self._class_stack):
            return None
        receiver_name = self._attribute_name(call_node.func.value)
        if not receiver_name or not receiver_name.startswith("self."):
            return None
        attr_source = normalize_source(self.instance_attrs.get(
            (self._class_stack[-1], receiver_name)))
        if isinstance(attr_source, CallResult):
            attr_source = normalize_source(
                attr_source.result_source or attr_source.callee)
        if not isinstance(attr_source, str):
            return None
        attr_top = self.symbols.get_top(attr_source) or attr_source
        if attr_top in (None, "", "local", "python", "unknown"):
            return None
        return _match_result_item_owner(attr_top, call_node.func.attr)

    ## Resolve the result owner of a protocol-dispatched NumPy ufunc.
    #
    #  Pandas and NumPy inputs retain their owner.  Python literals and
    #  containers produce NumPy results.  An unresolved or other import-backed
    #  receiver remains unknown because array protocols may override dispatch.
    #  @param call_node Ufunc call expression.
    #  @param func_top Resolved callable owner.
    #  @param func_name Resolved function name.
    #  @return Owner string, structured deferred source, UnknownSource, or
    #  None when not applicable.
    def _receiver_preserving_result_owner(self, call_node, func_top,
                                          func_name):
        if (func_top != "numpy"
                or func_name not in _RECEIVER_PRESERVE_UFUNCS
                or not call_node.args):
            return None
        arg_tops = []
        deferred_sources = []
        for argument in call_node.args:
            arg_top = self._expr_receiver_top(argument)
            if (_container_kind(argument) is not None
                    or isinstance(argument, ast.Constant)):
                arg_top = "python"
            if arg_top in (None, "", "local", "unknown"):
                dependency = self._parameter_dependency_source(
                    argument, expression_context=True)
                dependency = normalize_source(dependency)
                if dependency is None or isinstance(
                        dependency, UnknownSource):
                    return UnknownSource(
                        "receiver-preserving ufunc result")
                deferred_sources.append(dependency)
            else:
                deferred_sources.append(arg_top)
            arg_tops.append(arg_top)

        if any(top in (None, "", "local", "unknown")
               for top in arg_tops):
            return DerivedResult(
                "receiver_preserving_ufunc",
                tuple(deferred_sources),
                func_name,
            )

        external = set(top for top in arg_tops if top != "python")
        if not external:
            return "numpy"
        if len(external) == 1:
            owner = next(iter(external))
            if owner in ("pandas", "numpy"):
                return owner
        return UnknownSource("receiver-preserving ufunc result")

    ## Check whether a call expression is a known library-to-library conversion.
    #
    #  Unwraps trailing attribute chains (e.g. data.to_numpy().T) to find
    #  the inner conversion call, then looks up (source_library, method).
    #  @param value_node The RHS expression node.
    #  @return The conversion target library name, or None.
    def _resolve_conversion_target(self, value_node):
        # Unwrap trailing attribute chain: data.to_numpy().T → data.to_numpy()
        call_node = value_node
        while isinstance(call_node, ast.Attribute) and isinstance(call_node.value, (ast.Call, ast.Attribute)):
            call_node = call_node.value

        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Attribute):
            # Method call: data.to_numpy() → conversion if method in table.
            method = call_node.func.attr
            receiver = call_node.func.value
            if isinstance(receiver, ast.Name):
                receiver_top = self._receiver_top(receiver.id)
                if receiver_top and receiver_top not in ("local", "python", "unknown", ""):
                    conv = _CONVERSION_METHOD_TARGETS.get((receiver_top, method))
                    if conv:
                        return conv
            # Not a known conversion method — fall through to check
            # function-call return types (e.g. np.log, cdist).

        if isinstance(call_node, ast.Attribute):
            # Bare attribute read: data.values → conversion if attr in table.
            # Bare method references (data.to_numpy without call) are NOT
            # conversions — saving a method object does not change the result type.
            attr_name = call_node.attr
            receiver = call_node.value
            if isinstance(receiver, ast.Name):
                receiver_top = self._receiver_top(receiver.id)
                if receiver_top and receiver_top not in ("local", "python", "unknown", ""):
                    return _CONVERSION_ATTRIBUTE_TARGETS.get((receiver_top, attr_name))
            return None

        # 1.0.5 P1: function-call return type.  cdist(...) → numpy,
        # receiver-preserving ufunc np.log(pd.Series) → pandas.
        if isinstance(call_node, ast.Call):
            func_top, func_name = self._resolve_func_top(call_node.func)
            if func_top and func_top not in ("local", "python", "unknown", ""):
                # Check explicit result-owner map (cdist → numpy).
                ret = _match_result_owner(func_top, func_name)
                if ret:
                    return ret
                preserved = self._receiver_preserving_result_owner(
                    call_node, func_top, func_name)
                if preserved is not None:
                    return preserved
        return None

    ## Shared assignment pipeline: pending targets, trace RHS, visit, call_assign_funcs.
    #
    #  @param node The Assign or AnnAssign AST node.
    #  @param target_names Flat list of target name strings.
    #  @param field_target_names Instance-field targets used for bounded
    #  expression-owner propagation.
    #  @return The traced RHS source (right-hand value).
    def _visit_assignment(self, node, target_names,
                          field_target_names=None):
        imported_call = self._imported_call_result_source(node.value)
        pending_targets = list(target_names)
        pending_targets.extend(field_target_names or [])
        if pending_targets and isinstance(node.value, ast.Call):
            self._pending_call_targets_by_node[
                id(node.value)] = pending_targets

        result_item_owner = None
        if (isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Call)):
            result_item_owner = self._resolve_call_result_item_owner(
                node.value.value)
        right = (result_item_owner
                 or self._parameter_dependency_source(node.value)
                 or self.trace_source(node.value))
        right_norm = normalize_source(right)
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(right_norm, InstanceMethod)
                and isinstance(
                    normalize_source(right_norm.receiver), CallResult)
                and right_norm.receiver.result_source is None
                and _has_result_owner_contract(right_norm.method)):
            right = CallResult(
                right_norm,
                display_name=ast.unparse(node.value.func),
                call_lineno=node.value.lineno,
                call_col_offset=node.value.col_offset,
                result_source=DerivedResult(
                    "method_result",
                    (right_norm,),
                    node.value.func.attr,
                ),
            )
        # 1.0.5 P1: if the RHS is a known conversion call
        # (e.g. data.to_numpy()), override the bound source
        # so subsequent calls on the target use the post-conversion
        # library.  Also handles data.to_numpy().T chains.
        conversion = self._resolve_conversion_target(node.value)
        if conversion:
            right_norm = normalize_source(right)
            if isinstance(right_norm, CallResult):
                right = CallResult(
                    right_norm.callee,
                    display_name=right_norm.display_name,
                    call_lineno=right_norm.call_lineno,
                    call_col_offset=right_norm.call_col_offset,
                    result_source=conversion,
                )
            else:
                right = conversion
        # Preserve an owner proven by same-scope operator operands.  This
        # covers both instance fields and ordinary locals, but never replaces
        # parameter-backed dataflow: parameters are resolved from exact
        # project call edges in the cross-file pass.
        if isinstance(node.value, (ast.BinOp, ast.UnaryOp)):
            expression_top = self._expr_receiver_top(node.value)
            right_norm = normalize_source(right)
            has_parameter = self._source_contains_parameter(right_norm)
            if (not has_parameter
                    and expression_top not in (
                        None, "", "local", "python", "unknown")):
                if (right_norm in (
                        None, "", "local", "unknown")
                        or isinstance(
                            right_norm, (UnknownSource, DerivedResult))):
                    right = expression_top
            elif not has_parameter:
                external_tops = {
                    top for top in self._operator_operand_tops(node.value)
                    if top not in (
                        None, "", "local", "python", "unknown")
                }
                if len(external_tops) > 1:
                    right = UnknownSource(
                        "conflicting operator result owners")
        # 1.0.5 P0: visit RHS before binding targets
        right_norm = normalize_source(right)
        local_receiver = (self._returned_local_class_name(right_norm.receiver)
                          if isinstance(right_norm, InstanceMethod) else None)
        if (isinstance(node.value, ast.Call) and local_receiver is not None
                and right_norm.method in self.class_methods[local_receiver]):
            right = CallResult(
                right, display_name=ast.unparse(node.value.func),
                call_lineno=node.value.lineno,
                call_col_offset=node.value.col_offset,
                result_source=right)
        self.generic_visit(node)
        # Assignment-call metadata is flow-sensitive too.  Keep the previous
        # binding visible while nested RHS calls collect their argument
        # sources, then invalidate or replace it before binding the LHS.
        for name in target_names:
            key = (id(self.current_scope()), name)
            if imported_call is None:
                self._assigned_call_sources.pop(key, None)
            else:
                self._assigned_call_sources[key] = imported_call
        # 1.0.5 P0: call_assign_funcs after generic_visit
        value_node = node.value
        # Unwrap trailing attributes for call_assign_funcs:
        # data = data.to_numpy().T  →  extract data.to_numpy
        while isinstance(value_node, ast.Attribute):
            value_node = value_node.value
        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Attribute):
            func_full = self._attribute_name(value_node.func)
            if func_full:
                for name in target_names:
                    self.call_assign_funcs[name] = func_full
        return right

    ## Resolve a class identity through already defined local return summaries.
    #  @param source Receiver value, not the name of the producing method.
    #  @param seen Local return-summary cycle guard.
    #  @return Local class name or None without uniform constructor evidence.
    def _returned_local_class_name(self, source, seen=None):
        source = normalize_source(source)
        seen = set(seen or ())
        if isinstance(source, SourceSet):
            names = [self._returned_local_class_name(item, seen)
                     for item in source.sources]
            return (names[0] if names and names[0] is not None
                    and all(name == names[0] for name in names) else None)
        if isinstance(source, CallResult) and isinstance(source.callee, str):
            callee = source.callee
            if callee in self.class_methods:
                return callee
            if callee not in seen:
                seen.add(callee)
                return self._returned_local_class_name(
                    self.return_sources.get(callee), seen)
        if isinstance(source, str) and source in self.class_methods:
            return source
        return None

    ## Visit an AnnAssign node (x: T = expr) with RHS-before-target ordering.
    #
    #  Same contract as visit_Assign: visit the RHS value before binding
    #  the target symbol so nested RHS calls use the pre-assignment state.
    #  @param node The AnnAssign AST node.
    def visit_AnnAssign(self, node):
        if node.value is None:
            return
        mapping_value = self._mapping_facts.value(node.value)
        assignment_container_kind, assignment_item_kind = (
            self._expression_container_shape(node.value))
        assignment_item_fields = self._expression_container_item_fields(
            node.value)
        assignment_container_kind = assignment_container_kind or None
        assignment_item_kind = assignment_item_kind or None
        targets = []
        if isinstance(node.target, ast.Name):
            targets.append(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            name = self._attribute_name(node.target)
            if name and name.startswith("self."):
                targets.append(name)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    targets.append(elt.id)
        field_targets = [
            name for name in targets if name.startswith("self.")]
        right = self._visit_assignment(node, targets, field_targets)
        if isinstance(node.target, ast.Attribute):
            self._invalidate_attribute_tuple_item_sources(node.target)
        self._record_local_constructor_fields(node, targets)
        callable_keys = {}
        if isinstance(node.value, ast.Lambda):
            lambda_result = right or UnknownSource("lambda result")
            lambda_key = self._local_lambda_key(node.value)
            for name in targets:
                callable_keys[name] = lambda_key
            self.return_sources[lambda_key] = lambda_result
            right = "local"

        if right:
            right_norm = normalize_source(right)
            if isinstance(node.target, ast.Name):
                if isinstance(right, str) and right == node.target.id:
                    pass  # skip self-assign
                else:
                    self._bind_target_name(
                        node.target.id, right, node.target,
                        container_kind=assignment_container_kind or "",
                        container_item_kind=assignment_item_kind or "",
                        callable_key=callable_keys.get(node.target.id, ""),
                        container_item_fields=assignment_item_fields)
            elif isinstance(node.target, ast.Attribute):
                name = self._attribute_name(node.target)
                attr_name = name if name and name.startswith("self.") else (
                    self._instance_attribute_target_name(name))
                if attr_name:
                    self._bind_target_name(
                        attr_name, right, node.target,
                        container_kind=assignment_container_kind or "",
                        container_item_kind=assignment_item_kind or "",
                        container_item_fields=assignment_item_fields)
            elif isinstance(node.target, (ast.Tuple, ast.List)):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self._bind_target_name(elt.id, right, elt)
        else:
            if isinstance(node.target, ast.Name):
                self._bind_target_name(
                    node.target.id, 'local', node.target,
                    container_kind=assignment_container_kind or "",
                    container_item_kind=assignment_item_kind or "",
                    callable_key=callable_keys.get(node.target.id, ""),
                    container_item_fields=assignment_item_fields)
            elif isinstance(node.target, ast.Attribute):
                name = self._attribute_name(node.target)
                if name and name.startswith("self."):
                    self._bind_target_name(
                        name, 'local', node.target,
                        container_kind=assignment_container_kind or "",
                        container_item_kind=assignment_item_kind or "",
                        container_item_fields=assignment_item_fields)
            elif isinstance(node.target, (ast.Tuple, ast.List)):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self._bind_target_name(elt.id, 'local', elt)

        self._bind_mapping_value(node.target, mapping_value)

    ## Reject identities affected by augmented assignment or deletion.
    #  @param node AugAssign AST node.
    def visit_AugAssign(self, node):
        self._mapping_facts.escape(node.target)
        self._mapping_facts.escape(node.value)
        self.generic_visit(node)
        self._bind_mapping_value(node.target, None)

    ## Invalidate mapping aliases when a binding or item is deleted.
    #  @param node Delete AST node.
    def visit_Delete(self, node):
        for target in node.targets:
            self._mapping_facts.escape(target)
            self._bind_mapping_value(target, None)
        self.generic_visit(node)

    ## Do not reuse mapping identities across an unsupported walrus rebind.
    #  @param node NamedExpr AST node.
    def visit_NamedExpr(self, node):
        self.generic_visit(node)
        self._bind_mapping_value(node.target, None)

    ## --- API call detection ---

    ## Resolve the base of an API call for origin tracking.
    #
    #  Tries getattr(), import_module(), method resolution, and
    #  call-lookup receiver resolution in order.
    #  @param node The Call AST node.
    #  @return Base symbol, structured tuple, or None.
    def _resolve_call_base_for_api(self, node):
        # P0 (1.0.5): bare getattr() builtin calls must be classified as
        # python, not traced through the argument's provenance.  Only
        # trace through obj.getattr("name") style calls where getattr is
        # accessed as an attribute on a receiver object.
        if self._is_getattr_call(node) and not isinstance(node.func,
                                                          ast.Name):
            if self._literal_str(node.args[1]) is not None:
                g = self.trace_source(node.args[0])
                if g is not None:
                    return g
        if self._is_import_module_call(node):
            im = self._resolve_import_module_trace(node)
            if im is not None:
                return im
        base = self._resolve_methods(node)
        if base is not None:
            return base
        # 1.0.5 P1+: defaultdict(list) item kind — d[k].append(v) where
        # d = defaultdict(list) has item kind "list".
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Subscript):
                sub = node.func.value
                if isinstance(sub.value, ast.Name):
                    item_kind = self._lookup_container_kind(
                        sub.value.id, item=True)
                    if item_kind is not None:
                        return InstanceMethod(sub.value.id, node.func.attr)
        ## For chained calls (A().B()), resolve via the inner call's
        ## return source so the outer call traces to the correct library.
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            inner_source = self.trace_source(node.func.value)
            if isinstance(inner_source, str):
                rs = self.return_sources.get(inner_source)
                if rs is not None:
                    return rs
            if isinstance(inner_source, CallResult):
                if inner_source.result_source is not None:
                    return inner_source.result_source
                rs = self.return_sources.get(inner_source.callee)
                if rs is not None:
                    return rs
            if isinstance(inner_source, InstanceMethod):
                if inner_source.parameter_scope:
                    return InstanceMethod(
                        inner_source,
                        node.func.attr,
                        parameter_scope=inner_source.parameter_scope,
                        parameter_name=inner_source.parameter_name,
                    )
            if isinstance(inner_source, SourceSet):
                return inner_source
            # 1.0.5 P1: library-function return types for chained
            # calls (cdist(...).argmin(), np.log(s).diff()).
            inner_call = node.func.value
            func_top, func_name = self._resolve_func_top(inner_call.func)
            if func_top and func_top not in ("local", "python", "unknown", ""):
                ret = _match_result_owner(func_top, func_name)
                if ret:
                    return ret
                preserved = self._receiver_preserving_result_owner(
                    inner_call, func_top, func_name)
                if preserved is not None:
                    return preserved
        call_lookup_base = self.get_base(node, call_lookup=True)
        if call_lookup_base is not None:
            return call_lookup_base
        return self.get_base(node.func)

    ## Collect all prefix calls in a chained call expression.
    #
    #  For a.b().c().d(), returns [a.b(), c(), d()] in call order.
    #  @param node The outermost Call AST node.
    #  @return List of Call nodes from outermost to innermost chain, reversed.
    def _chained_prefix_calls(self, node):
        if not isinstance(node, ast.Call):
            return []
        out = []
        cur = node
        while isinstance(cur, ast.Call):
            out.append(cur)
            f = cur.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
                cur = f.value
            else:
                break
        out.reverse()
        return out

    ## Record a single API call with its resolved top-level origin.
    #  @param node The Call AST node.
    def _one_api_call(self, node):
        if id(node) in self._seen_api_call_ids:
            return
        self._seen_api_call_ids.add(id(node))
        api_string = self.get_call(node)
        func_name, parameters = self._get_call_parts(node)
        base = self._resolve_call_base_for_api(node)
        if not base:
            return

        ## Record CallEdge fact for Phase 7B-full call graph.
        if self._caller_stack:
            caller = self._caller_stack[-1]
            ## Collect receiver source for obj.method() calls.
            receiver_source = None
            if isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if (isinstance(receiver, ast.Name)
                        and receiver.id in ("self", "cls")):
                    receiver_source = self.get_base(receiver)
                else:
                    receiver_source = self._call_edge_argument_source(
                        receiver)
            ## Collect arg sources.  Ordinary positional and keyword
            #  arguments stay in the existing maps; starred expansions are
            #  retained separately so cross-file binding can respect the
            #  callee signature instead of treating a pack as one value.
            arg_sources = {"pos": {}, "kw": {}}
            protocol_arg_sources = {"pos": {}, "kw": {}}
            iterable_arg_sources = {"pos": {}, "kw": {}}
            star_arg_sources = {}
            positional_index = 0
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    star_src = self._call_edge_argument_source(arg.value)
                    if star_src is not None:
                        star_arg_sources[positional_index] = star_src
                    positional_index = None
                    continue
                arg_src = self._call_edge_argument_source(arg)
                if arg_src is not None:
                    if positional_index is not None:
                        arg_sources["pos"][positional_index] = arg_src
                protocol_src = self._call_edge_protocol_source(arg)
                if protocol_src is not None:
                    if positional_index is not None:
                        protocol_arg_sources["pos"][positional_index] = protocol_src
                iterable_src = self._call_edge_iterable_source(arg)
                if iterable_src is not None:
                    if positional_index is not None:
                        iterable_arg_sources["pos"][positional_index] = iterable_src
                if positional_index is not None:
                    positional_index += 1
            star_kwarg_sources = []
            for kw in getattr(node, "keywords", []) or []:
                if kw.arg is None:
                    star_src = self._call_edge_argument_source(kw.value)
                    if star_src is not None:
                        star_kwarg_sources.append(star_src)
                    continue
                arg_src = (
                    self._call_edge_argument_source(kw.value)
                    if kw.arg else None)
                if arg_src is not None and kw.arg:
                    arg_sources["kw"][kw.arg] = arg_src
                protocol_src = (
                    self._call_edge_protocol_source(kw.value)
                    if kw.arg else None)
                if protocol_src is not None and kw.arg:
                    protocol_arg_sources["kw"][kw.arg] = protocol_src
                iterable_src = (
                    self._call_edge_iterable_source(kw.value)
                    if kw.arg else None)
                if iterable_src is not None and kw.arg:
                    iterable_arg_sources["kw"][kw.arg] = iterable_src
            ## Consume assigned_to only for the top-level RHS call.
            assigned = self._pending_call_targets_by_node.pop(id(node), [])
            callback_args = {}
            for index, arg in enumerate(node.args):
                callback_name = self._call_edge_callback_name(arg)
                if callback_name:
                    callback_args[index] = callback_name
            callback_bindings = []
            target_names = {
                kw.value.id for kw in getattr(node, "keywords", []) or []
                if kw.arg == "target" and isinstance(kw.value, ast.Name)
            }
            args_keyword = next(
                (kw.value for kw in getattr(node, "keywords", []) or []
                 if kw.arg == "args"), None)
            callback_source = self._call_edge_tuple_source(args_keyword)
            if callback_source is not None:
                callback_bindings = [
                    {"callback": target, "args": callback_source}
                    for target in sorted(target_names)
                ]
            edge = CallEdge(
                caller=caller,
                callee=base,
                callee_name=func_name or "",
                callee_source=self._call_edge_callee_source(node.func),
                receiver_source=receiver_source,
                arg_sources=arg_sources,
                star_arg_sources=star_arg_sources,
                star_kwarg_sources=star_kwarg_sources,
                callback_args=callback_args,
                callback_bindings=callback_bindings,
                protocol_arg_sources=protocol_arg_sources,
                iterable_arg_sources=iterable_arg_sources,
                assigned_to=assigned,
                call_lineno=node.lineno,
                call_col_offset=node.col_offset,
            )
            self.module_cg.edges.append(edge)
            mapping_value = self._mapping_facts.value(node.func)
            if mapping_value is not None and mapping_value.selected:
                self._mapping_edges.append((edge, mapping_value))

        if isinstance(node.func, ast.Name):
            direct_name = node.func.id
        else:
            direct_name = None

        scope_name = ""
        cs = self.current_scope()
        if cs.kind != SCOPE_MODULE:
            scope_name = cs.name
        loc = {
            'func_name': func_name,
            'parameters': parameters,
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'end_lineno': getattr(node, 'end_lineno', 0) or 0,
            'end_col_offset': getattr(node, 'end_col_offset', 0) or 0,
            'scope_name': scope_name,
        }
        if isinstance(normalize_source(base), SuperMethod):
            # Snapshot the class-definition import path, before later
            # rebinding or another same-named class can replace the evidence.
            loc['super_base_path'] = None
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            if (isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Name)
                    and receiver.func.id == 'super'
                    and not receiver.args and not receiver.keywords
                    and _is_unshadowed_builtin_call(self, receiver)
                    and cs.kind == SCOPE_FUNCTION
                    and cs.parent is not None and cs.parent.kind == SCOPE_CLASS
                    and self._super_base_path_stack):
                base_path, decorator_module = self._super_base_path_stack[-1]
                loc['super_base_path'] = base_path
                loc['super_decorator_module'] = decorator_module
        # 1.0.5 P0: snapshot call_assign_funcs for dotted calls so
        # cross-file _resolve_func_name reads the pre-assignment
        # state, not the final map that may include later
        # reassignments.
        if func_name and '.' in func_name:
            first = func_name.split('.')[0]
            loc['call_assign_func'] = self.call_assign_funcs.get(first)
            binding = self.current_scope().lookup(
                first, skip_parent_classes=True)
            if (binding is not None
                    and binding.binding_kind == "import"
                    and binding.scope_kind != SCOPE_MODULE):
                loc["call_import_source"] = (
                    self._import_binding_sources.get(
                        self._binding_key(binding))
                    or binding.source
                )

        # The callable identity of an unshadowed builtin is independent of
        # the object it returns.  Record it before return-value provenance can
        # introduce an unrelated same-name assignment from another scope.
        if direct_name and _is_unshadowed_builtin_call(self, node):
            record = {
                'api': api_string,
                'top': 'python',
                'chain': ['python'],
                'base': direct_name,
                'direct_name_callee': direct_name,
            }
            record.update(loc)
            self.api_calls.append(record)
            self._collect_call_site(api_string, func_name, parameters,
                                    direct_name, loc)
            return

        # A concrete receiver kind determines the callable owner even when
        # legacy base resolution represents the receiver as a plain string.
        # This covers names, self attributes, and homogeneous subscript items.
        receiver_kind = self._call_receiver_container_kind(node)
        method_name = (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if (receiver_kind is not None
                and _has_builtin_shape_method(receiver_kind, method_name)):
            loc["receiver_container_kind"] = receiver_kind
            record = {
                'api': api_string,
                'top': 'python',
                'chain': ['python'],
                'base': base,
                'direct_name_callee': direct_name,
            }
            record.update(loc)
            self.api_calls.append(record)
            self._collect_call_site(api_string, func_name, parameters,
                                    base, loc)
            return

        if isinstance(base, UnknownSource):
            record = {
                'api': api_string,
                'top': 'unknown',
                'chain': ['unknown'],
                'base': base,
                'direct_name_callee': direct_name,
            }
            record.update(loc)
            self.api_calls.append(record)
            self._collect_call_site(api_string, func_name, parameters,
                                    base, loc)
            return

        if isinstance(base, CallResult):
            # Resolve top through the callee so s.get() shows 'requests'
            # instead of 'requests()' when s = Session().
            callee = base.callee
            ## 1.0.5 P2: explicit result_source carries result-object ownership.
            #  When set, it overrides callee-based tracing — the callable's
            #  identity is determined by what the called function returns,
            #  not who was called.
            rs_explicit = getattr(base, 'result_source', None)
            if rs_explicit is not None:
                if isinstance(rs_explicit, UnknownSource):
                    top = "unknown"
                elif isinstance(rs_explicit, PythonShape):
                    top = "python"
                elif rs_explicit == "python":
                    top = "python"
                elif is_structured_source(rs_explicit):
                    # Structured: defer to cross_file for resolution.
                    # In single-file, use source_display as placeholder.
                    top = source_display(base)
                else:
                    # String source — direct ownership.
                    top = str(rs_explicit)
            elif isinstance(callee, str):
                rs = self.return_sources.get(callee)
                if rs is not None:
                    resolved = normalize_source(rs)
                    if isinstance(resolved, CallResult):
                        inner_callee = resolved.callee
                        if isinstance(inner_callee, str):
                            callee = inner_callee
                    elif isinstance(resolved, SourceSet):
                        top = source_display(base)
                        chain = [top]
                        record = {
                            'api': api_string,
                            'top': top,
                            'chain': chain,
                            'base': base,
                            'direct_name_callee': direct_name,
                        }
                        record.update(loc)
                        self.api_calls.append(record)
                        self._collect_call_site(api_string, func_name, parameters,
                                                base, loc)
                        return
                top = self.symbols.get_top(callee) or source_display(base)
            else:
                top = source_display(base)
            chain = [source_display(base)]
            record = {
                'api': api_string,
                'top': top,
                'chain': chain,
                'base': base,
                'direct_name_callee': direct_name,
            }
            record.update(loc)
            self.api_calls.append(record)
            self._collect_call_site(api_string, func_name, parameters,
                                    base, loc)
            return

        if isinstance(base, tuple) or isinstance(base, (ContainerItem, ContainerIter, InstanceMethod, SuperMethod, SourceSet)):
            # Unresolved compare receiver: owner cannot be determined.
            # Emit as unknown so the call is collected but not
            # misattributed to local.
            if (isinstance(base, InstanceMethod)
                    and isinstance(base.receiver, str)
                    and base.receiver == "__unresolved_compare__"):
                display = "unknown"
                chain = ["unknown"]
                record = {
                    'api': api_string,
                    'top': display,
                    'chain': chain,
                    'base': base,
                    'direct_name_callee': direct_name,
                }
                record.update(loc)
                self.api_calls.append(record)
                self._collect_call_site(api_string, func_name, parameters,
                                        base, loc)
                return

            display = source_display(base)
            if isinstance(base, InstanceMethod) and isinstance(base.receiver, str):
                top_from_receiver = self.symbols.get_top(base.receiver)
                # 1.0.5 P1: builtin container methods on receivers
                # whose container kind is known.  The receiver may be
                # local via get_top, scope binding, or item kind
                # (defaultdict(list) — receiver traces to collections
                # but item kind is list).
                #
                rec_local = (top_from_receiver == "local")
                if not rec_local:
                    binding = self.current_scope().lookup(base.receiver)
                    if (binding is not None
                            and binding.source == "local"):
                        rec_local = True
                # Use the exact call receiver shape to distinguish
                # d[k].append() item metadata from d.append(). Lexical
                # binding metadata prevents same-name scope leakage.
                kind = self._call_receiver_container_kind(node)
                if rec_local or kind is not None:
                    if (kind is not None
                            and _has_builtin_shape_method(kind, base.method)):
                        display = "python"
                        # Record kind at call-site time so cross-file
                        # phase doesn't see later invalidation.
                        loc["receiver_container_kind"] = kind
                    elif rec_local and not base.parameter_scope:
                        display = "local"
                    else:
                        display = display  # keep source_display default
            chain = [display] if display else []
            record = {
                'api': api_string,
                'top': display,
                'chain': chain,
                'base': base,
                'direct_name_callee': direct_name,
            }
            record.update(loc)
            self.api_calls.append(record)
            self._collect_call_site(api_string, func_name, parameters,
                                    base, loc)
            return

        # Handle a lexical binding that resolves to a local value.
        if base == "local":
            record = {
                'api': api_string,
                'top': 'local',
                'chain': ['local'],
                'base': 'local',
                'direct_name_callee': direct_name,
            }
            record.update(loc)
            self.api_calls.append(record)
            self._collect_call_site(api_string, func_name, parameters,
                                    base, loc)
            return

        top = self.symbols.get_top(base)
        if not top:
            return

        record = {
            'api': api_string,
            'top': top,
            'chain': self.symbols.get_chain(base),
            'base': base,
            'direct_name_callee': direct_name,
        }
        record.update(loc)
        self.api_calls.append(record)
        self._collect_call_site(api_string, func_name, parameters,
                                base, loc)

    ## Collect a CallSite from the raw call data.
    #  @param expression Full call expression.
    #  @param func_name Function part.
    #  @param parameters Arguments string.
    #  @param base Base symbol or source.
    #  @param loc Dict with lineno/col_offset/end_lineno/end_col_offset.
    def _collect_call_site(self, expression, func_name, parameters,
                           base, loc):
        scope_name = ""
        cs = self.current_scope()
        if cs.kind != SCOPE_MODULE:
            scope_name = cs.name
        self.call_site_objects.append(CallSite(
            expression=expression,
            func_name=func_name,
            parameters=parameters,
            base=base,
            module_name=self.module_name or "",
            file_path=getattr(self, '_file_path', ""),
            lineno=loc.get('lineno', 0),
            col_offset=loc.get('col_offset', 0),
            end_lineno=loc.get('end_lineno', 0),
            end_col_offset=loc.get('end_col_offset', 0),
            scope_name=scope_name,
        ))

    ## Record verified owner evidence supplied to callback parameters.
    #
    #  @param node The call expression accepting a callback.
    def _collect_callback_parameter_sources(self, node):
        func_top, func_name = self._resolve_func_top(node.func)
        if not func_top or not func_name:
            return
        for (owner, name, callback_index, parameter_index), contract in (
                _CALLBACK_PARAMETER_OWNER_CONTRACTS.items()):
            if (name != func_name
                    or not (func_top == owner
                            or func_top.startswith(owner + "."))
                    or callback_index >= len(node.args)):
                continue
            callback = node.args[callback_index]
            callback_key = self._value_source(callback)
            if not isinstance(callback_key, str):
                continue
            bare_key = callback_key.rsplit(".", 1)[-1]
            params = (self.function_params.get(callback_key)
                      or self.function_params.get(bare_key, []))
            if parameter_index >= len(params):
                continue
            parameter = params[parameter_index]
            self.parameter_sources.setdefault(
                (callback_key, parameter), []).append(contract[0])

    ## Return (func_str, args_str) tuple for a Call node.
    #  @param node The Call AST node.
    #  @return Tuple of (function expression, arguments string).
    def _get_call_parts(self, node):
        func_str = ast.unparse(node.func)
        parts = [ast.unparse(a) for a in node.args]
        if node.keywords:
            for kw in node.keywords:
                if kw.arg:
                    parts.append(f"{kw.arg}={ast.unparse(kw.value)}")
                else:
                    parts.append(f"**{ast.unparse(kw.value)}")
        args_str = ", ".join(parts)
        return func_str, args_str

    ## Reconstruct a call expression as a string.
    #  @param node The Call AST node.
    #  @return String representation like "func(arg1, arg2, kw=val)".
    def get_call(self, node):
        func_str, args_str = self._get_call_parts(node)
        return f"{func_str}({args_str})"

    ## Visit a Call node and record API calls from its chained prefix calls.
    #  @param node The Call AST node.
    def visit_Call(self, node):
        self._mapping_facts.call(node)
        self._collect_argparse_destination(node)
        self._collect_callback_parameter_sources(node)
        self._record_container_append_shape(node)
        for sub in self._chained_prefix_calls(node):
            self._one_api_call(sub)
        if isinstance(node.func, ast.Name) and node.func.id in self.defined_functions:
            arg_sources = []
            protocol_arg_sources = []
            for arg in node.args:
                if isinstance(arg, ast.Attribute):
                    name = self._attribute_name(arg)
                    arg_sources.append(
                        name if name else self._call_edge_argument_source(arg))
                else:
                    arg_sources.append(self._call_edge_argument_source(arg))
                protocol_arg_sources.append(
                    self._call_edge_protocol_source(arg))
            self.call_sites.setdefault(node.func.id, []).append({
                "module": self.module_name,
                "args": arg_sources,
                "protocol_args": protocol_arg_sources,
                "lineno": node.lineno,
                "col_offset": node.col_offset,
            })
        elif isinstance(node.func, ast.Name) and node.func.id in self.class_methods:
            arg_sources = []
            protocol_arg_sources = []
            for arg in node.args:
                if isinstance(arg, ast.Attribute):
                    name = self._attribute_name(arg)
                    arg_sources.append(
                        name if name else self._call_edge_argument_source(arg))
                else:
                    arg_sources.append(self._call_edge_argument_source(arg))
                protocol_arg_sources.append(
                    self._call_edge_protocol_source(arg))
            self.call_sites.setdefault(node.func.id + ".__init__", []).append({
                "module": self.module_name,
                "args": arg_sources,
                "protocol_args": protocol_arg_sources,
                "lineno": node.lineno,
                "col_offset": node.col_offset,
            })
        self.generic_visit(node)

    ## Record fields of a literal element appended to a known list.
    #  @param node Append call AST node.
    def _record_container_append_shape(self, node):
        if (not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr != "append"
                or len(node.args) != 1):
            return
        receiver = node.func.value
        receiver_kind, _ = self._expression_container_shape(receiver)
        if receiver_kind != "list":
            return

        ## Preserve a homogeneous owner fact for later iteration.  This is
        # based on the appended value's traced source, not on append itself.
        # Any unresolved or conflicting append invalidates the fact.
        def record_item_source(binding, item_source):
            if binding is None or binding.container_kind != "list":
                return
            key = self._binding_key(binding)
            current = self._iterated_append_sources.get(key)
            current_norm = normalize_source(current)
            current_owner = (
                self._structured_source_owner_top(current_norm)
                if current_norm is not None else None)
            item_owner = self._structured_source_owner_top(item_source)
            if isinstance(current_norm, UnknownSource):
                current_display = current_norm.display
                if current_display == "empty iterable":
                    current_owner = None
            if item_owner is None:
                self._iterated_append_sources[key] = UnknownSource(
                    "unresolved appended item")
            elif current_owner is None or current_owner == item_owner:
                self._iterated_append_sources[key] = item_owner
            else:
                self._iterated_append_sources[key] = UnknownSource(
                    "conflicting iterable items")

        def tuple_item_source(value):
            """Return exact field evidence for one appended tuple/list."""
            if not isinstance(value, (ast.Tuple, ast.List)):
                return None
            items = []
            for field in value.elts:
                shape = self._expression_python_shape(field)
                if shape is not None:
                    items.append(shape)
                    continue
                source = normalize_source(
                    self._call_edge_argument_source(field))
                if source is None or isinstance(source, UnknownSource):
                    return None
                items.append(source)
            return TupleSource(tuple(items))

        item_source = normalize_source(self.trace_source(node.args[0]))
        if isinstance(receiver, ast.Name):
            binding = self.current_scope().lookup(
                receiver.id, skip_parent_classes=True)
            record_item_source(binding, item_source)
            tuple_source = tuple_item_source(node.args[0])
            if binding is not None:
                key = self._binding_key(binding)
                if key in self._iterated_append_tuple_conflicts:
                    pass
                elif tuple_source is None:
                    self._iterated_append_tuple_sources.pop(key, None)
                    self._iterated_append_tuple_conflicts.add(key)
                else:
                    previous = self._iterated_append_tuple_sources.get(key)
                    if previous is None or previous == tuple_source:
                        self._iterated_append_tuple_sources[key] = tuple_source
                    else:
                        self._iterated_append_tuple_sources.pop(key, None)
                        self._iterated_append_tuple_conflicts.add(key)
        elif isinstance(receiver, ast.Attribute):
            key = self._attribute_container_key(receiver)
            tuple_source = tuple_item_source(node.args[0])
            if key is not None:
                if key in self._attribute_append_tuple_conflicts:
                    pass
                elif tuple_source is None:
                    self._attribute_append_tuple_sources.pop(key, None)
                    self._attribute_append_tuple_conflicts.add(key)
                else:
                    previous = self._attribute_append_tuple_sources.get(key)
                    if (previous is None
                            or previous == tuple_source
                            or self._tuple_item_owners_match(
                                previous, tuple_source)):
                        self._attribute_append_tuple_sources[key] = tuple_source
                    else:
                        self._attribute_append_tuple_sources.pop(key, None)
                        self._attribute_append_tuple_conflicts.add(key)

        ## Preserve a homogeneous item shape only when the appended value is
        # independently proven by syntax or an existing result contract.  An
        # unknown or conflicting append invalidates the binding fact so later
        # iteration cannot infer a receiver type from an incomplete list.
        item_shape = self._expression_python_shape(node.args[0])
        item_kind = (
            item_shape.kind
            if isinstance(item_shape, PythonShape) and item_shape.kind
            else "")

        def record_item_kind(binding, container_name):
            if binding is None or binding.container_kind != "list":
                return
            conflict_key = self._binding_key(binding)
            if conflict_key in self._container_item_kind_conflicts:
                return
            current = binding.container_item_kind or ""
            if not item_kind or (current and current != item_kind):
                binding.container_item_kind = ""
                self.container_item_kinds.pop(container_name, None)
                self._container_item_kind_conflicts.add(conflict_key)
                return
            binding.container_item_kind = item_kind
            self.container_item_kinds[container_name] = item_kind

        if isinstance(receiver, ast.Name):
            record_item_kind(
                self.current_scope().lookup(
                    receiver.id, skip_parent_classes=True),
                receiver.id)

        fields = self._literal_dict_field_shapes(node.args[0])
        if not fields:
            if isinstance(receiver, ast.Attribute):
                name = self._attribute_name(receiver)
                if name and name.startswith("self.") and self._class_stack:
                    key = (self._class_stack[-1], name)
                    if item_kind:
                        self.instance_attr_item_kinds[key] = item_kind
                    else:
                        self.instance_attr_item_kinds.pop(key, None)
            return
        if isinstance(receiver, ast.Name):
            binding = self.current_scope().lookup(
                receiver.id, skip_parent_classes=True)
            if binding is not None and binding.container_kind == "list":
                binding.container_item_fields = dict(fields)
            return
        if isinstance(receiver, ast.Attribute):
            name = self._attribute_name(receiver)
            if name and name.startswith("self.") and self._class_stack:
                self.instance_attr_item_fields[(
                    self._class_stack[-1], name)] = dict(fields)

    ## Visit an Attribute access node and record the top-level origin.
    #  @param node The Attribute AST node.
    def visit_Attribute(self, node):
        attr_string = ast.unparse(node)
        name = self._attribute_name(node)
        if name and name in self.symbols.direct:
            base = name
        else:
            base = self.get_base(node)
        if base:
            top = self.symbols.get_top(base)
            if top:
                self.attr_accesses.append({
                    'attr': attr_string,
                    'top': top,
                    'chain': self.symbols.get_chain(base)
                })
        self.generic_visit(node)

    ## Collect statically declared pytest parameter values.
    #
    #  pytest.mark.parametrize supplies concrete call-site-like evidence for
    #  test parameters even though pytest performs the invocation at runtime.
    #  Only literal parameter names and literal value sequences are accepted.
    #  @param node Function definition carrying decorators.
    #  @param qualname Qualified function name.
    #  @param params Declared function parameter names.
    def _collect_parametrize_sources(self, node, qualname, params):
        for decorator in node.decorator_list:
            if (not isinstance(decorator, ast.Call)
                    or not isinstance(decorator.func, ast.Attribute)
                    or decorator.func.attr != "parametrize"
                    or len(decorator.args) < 2):
                continue
            owner = self.get_base(decorator.func.value)
            owner_top = self.symbols.get_top(owner) if owner else None
            if owner != "pytest" and owner_top != "pytest":
                continue

            names_node = decorator.args[0]
            if (isinstance(names_node, ast.Constant)
                    and isinstance(names_node.value, str)):
                names = [part.strip()
                         for part in names_node.value.split(",")
                         if part.strip()]
            elif isinstance(names_node, (ast.List, ast.Tuple)):
                names = [
                    item.value for item in names_node.elts
                    if (isinstance(item, ast.Constant)
                        and isinstance(item.value, str))
                ]
            else:
                continue
            if not names or any(name not in params for name in names):
                continue

            values_node = decorator.args[1]
            if not isinstance(values_node, (ast.List, ast.Tuple)):
                continue
            for case in values_node.elts:
                case_nodes = (
                    list(case.elts)
                    if len(names) > 1 and isinstance(case, (ast.List, ast.Tuple))
                    else [case]
                )
                if len(case_nodes) != len(names):
                    continue
                for name, value_node in zip(names, case_nodes):
                    source = self._value_source(value_node)
                    if source is not None:
                        self.parameter_sources.setdefault(
                            (qualname, name), []).append(source)

    ## Return a kwargs parameter for a direct __dict__.update contract.
    #  @param node The constructor AST node.
    #  @return kwargs parameter name, or None when no direct contract exists.
    def _constructor_kwargs_parameter(self, node):
        if node.name != "__init__" or node.args.kwarg is None:
            return None
        kwargs_name = node.args.kwarg.arg
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if (not isinstance(child.func, ast.Attribute)
                    or child.func.attr != "update"
                    or not child.args
                    or not isinstance(child.args[0], ast.Name)
                    or child.args[0].id != kwargs_name):
                continue
            if (isinstance(child.func.value, ast.Attribute)
                    and self._attribute_name(child.func.value)
                    == "self.__dict__"):
                return kwargs_name
        return None

    ## Return one unambiguous wildcard-import owner, when available.
    #
    #  A wildcard import is still import evidence, but it does not identify
    #  the imported symbol.  It is therefore safe to use only when the file
    #  has one unique wildcard source.  Multiple wildcard sources remain
    #  unresolved rather than selecting one by name.
    #  @return Module name or None when wildcard evidence is ambiguous.
    def _unique_wildcard_import_owner(self):
        owners = []
        for owner in self.wildcard_modules:
            if not isinstance(owner, str) or not owner:
                continue
            top = owner.split(".", 1)[0]
            if top not in owners:
                owners.append(top)
        return owners[0] if len(owners) == 1 else None

    ## Return the lexical scope key for local instance field facts.
    #  @return Tuple identifying the current module, class, and function scope.
    def _local_instance_field_scope_key(self):
        scope = tuple(self._class_stack) + tuple(self._func_stack)
        return scope or ("<module>",)

    ## Return lexical scopes eligible for a local instance field lookup.
    #  A binding in the current scope blocks inherited field facts, including
    #  a function-local global rebinding. An unbound name may inherit facts
    #  from enclosing function, class, or module scope.
    #  @param root_name Root name of the instance expression.
    #  @return Scope keys ordered from innermost to outermost.
    def _local_instance_field_scope_keys(self, root_name):
        scope_key = self._local_instance_field_scope_key()
        keys = [scope_key]
        if (root_name in self.current_scope().bindings
                or root_name in self._local_instance_field_shadowed.get(
                    scope_key, set())):
            return keys
        if self._func_stack:
            for depth in range(len(self._func_stack) - 1, 0, -1):
                keys.append(tuple(self._class_stack)
                            + tuple(self._func_stack[:depth]))
        if keys[-1] != ("<module>",):
            keys.append(("<module>",))
        return keys

    ## Record fields established by a local kwargs-backed constructor call.
    #  Only direct keyword arguments are retained. Constructors without an
    #  explicit self.__dict__.update(kwargs) contract remain unresolved.
    #  @param node Assignment node.
    #  @param target_names Names assigned by the node.
    def _record_local_constructor_fields(self, node, target_names):
        scope_key = self._local_instance_field_scope_key()
        shadowed = self._local_instance_field_shadowed.setdefault(
            scope_key, set())
        for target_name in target_names:
            shadowed.add(target_name)
            for key in [key for key in self._local_instance_field_sources
                        if key[0] == scope_key and key[1] == target_name]:
                self._local_instance_field_sources.pop(key, None)
                self._local_instance_field_shapes.pop(key, None)
        if (not isinstance(node.value, ast.Call)
                or not isinstance(node.value.func, ast.Name)):
            return
        class_name = node.value.func.id
        if class_name not in self._constructor_kwargs_contracts:
            return
        keywords = [kw for kw in node.value.keywords if kw.arg is not None]
        for target_name in target_names:
            for keyword in keywords:
                source = self._value_source(keyword.value)
                if source is None:
                    source = self.trace_source(keyword.value)
                if source is not None:
                    self._local_instance_field_sources[
                        (scope_key, target_name, keyword.arg)] = source
                shape = self._expression_container_shape(keyword.value)
                if shape[0]:
                    self._local_instance_field_shapes[
                        (scope_key, target_name, keyword.arg)] = shape

    ## Visit a FunctionDef node and register it as a local definition.
    #  @param node The FunctionDef AST node.
    def _visit_function_def(self, node):
        """Common handler for FunctionDef and AsyncFunctionDef."""
        is_direct_method = bool(self._class_stack and not self._func_stack)
        receiver_name = ""
        if is_direct_method:
            decorator_names = {
                self._attribute_name(decorator)
                if isinstance(decorator, ast.Attribute)
                else decorator.id
                for decorator in node.decorator_list
                if isinstance(decorator, (ast.Name, ast.Attribute))
            }
            positional = (
                list(getattr(node.args, "posonlyargs", []))
                + list(node.args.args)
            )
            is_static = any(
                name == "staticmethod"
                or (isinstance(name, str)
                    and name.endswith(".staticmethod"))
                for name in decorator_names
            )
            if positional and not is_static:
                receiver_name = positional[0].arg
        self._class_receiver_stack.append(receiver_name)
        self.local.add(node.name)
        if self._class_stack:
            kwargs_name = self._constructor_kwargs_parameter(node)
            if kwargs_name is not None:
                self._constructor_kwargs_contracts[
                    self._class_stack[-1]] = kwargs_name
        ## Only module-level functions shadow builtins through bare-name
        ## calls.  Class methods are reachable only via self.<name>().
        if not self._class_stack:
            self.defined_functions.add(node.name)
        callable_key = self._local_callable_key(node.name)
        self._bind_target_name(
            node.name, "local", node, callable_key=callable_key)
        if not node.decorator_list and not is_direct_method:
            self.current_scope().bindings[node.name].mapping_value = (
                self._mapping_facts.callable(FunctionId(
                    self.module_name or "", callable_key)))
        params = []
        positional_params = []
        keyword_only_params = []
        vararg_name = ""
        kwarg_name = ""
        positional_nodes = (
            list(getattr(node.args, "posonlyargs", []))
            + list(node.args.args))
        defaults = {}
        positional_default_nodes = list(getattr(node.args, "defaults", []))
        if positional_default_nodes:
            positional_default_params = positional_nodes[-len(
                positional_default_nodes):]
            for arg, default_node in zip(
                    positional_default_params, positional_default_nodes):
                default_source = self._default_argument_source(
                    self._call_edge_argument_source(default_node))
                if default_source is not None:
                    defaults[arg.arg] = default_source
        for arg, default_node in zip(
                getattr(node.args, "kwonlyargs", []),
                getattr(node.args, "kw_defaults", [])):
            if default_node is None:
                continue
            default_source = self._default_argument_source(
                self._call_edge_argument_source(default_node))
            if default_source is not None:
                defaults[arg.arg] = default_source
        self.push_scope(SCOPE_FUNCTION, node.name)
        if receiver_name and receiver_name != "self":
            self._bind_target_name(receiver_name, "local", node, "parameter")
        for arg in positional_nodes:
            if arg.arg != receiver_name:
                params.append(arg.arg)
                positional_params.append(arg.arg)
                self._bind_target_name(arg.arg, "local", arg, "parameter")
        for arg in getattr(node.args, "kwonlyargs", []):
            if arg.arg != receiver_name:
                params.append(arg.arg)
                keyword_only_params.append(arg.arg)
                self._bind_target_name(arg.arg, "local", arg, "parameter")
        if getattr(node.args, "vararg", None) is not None and node.args.vararg.arg != "self":
            params.append(node.args.vararg.arg)
            vararg_name = node.args.vararg.arg
            self._bind_target_name(node.args.vararg.arg, "local", kind="parameter")
        if getattr(node.args, "kwarg", None) is not None and node.args.kwarg.arg != "self":
            params.append(node.args.kwarg.arg)
            kwarg_name = node.args.kwarg.arg
            self._bind_target_name(node.args.kwarg.arg, "local", kind="parameter")
        self.function_params[node.name] = params
        if self._class_stack:
            self.function_params[self._class_stack[-1] + "." + node.name] = params
        self._func_stack.append(node.name)
        ## Determine qualified name for call-graph facts.
        ## Use full _func_stack so nested functions get "outer.inner".
        if self._class_stack:
            qualname = self._class_stack[-1] + "." + ".".join(self._func_stack)
        else:
            qualname = ".".join(self._func_stack)
        self._collect_parametrize_sources(node, qualname, params)
        fid = FunctionId(self.module_name or "", qualname)
        self._caller_stack.append(fid)
        # Save and clear _global_names so each function independently
        # scopes global declarations.
        saved_globals = self._global_names
        self._global_names = set()
        self._mapping_local_names[id(self.current_scope())] = (
            mapping_bound_names(node.body))
        self.generic_visit(node)
        self._global_names = saved_globals
        self._caller_stack.pop()
        self._func_stack.pop()
        self.pop_scope()
        self._class_receiver_stack.pop()
        ## Collect FunctionSummary for Phase 7B-full facts.
        ## Use qualname so class methods don't share bare-name keys.
        func_returns = self.call_graph_return_sources.get(qualname)
        if func_returns is None:
            func_returns = self.return_sources.get(qualname)
        if func_returns is None and not self._class_stack:
            func_returns = self.return_sources.get(node.name)
        yield_values = self.call_graph_yield_sources.get(qualname, [])
        func_yields = (
            make_source_set(yield_values, origin="yield")
            if yield_values else None)
        return_values = self.call_graph_return_values.get(qualname)
        if not self._block_exits(node.body):
            return_values = make_source_set(
                [return_values, PythonShape("NoneType")]
                if return_values is not None else [PythonShape("NoneType")],
                origin="return")
        if func_yields is not None or isinstance(node, ast.AsyncFunctionDef):
            return_values = UnknownSource("deferred function result")
        local_assignments = {}
        fs = FunctionSummary(
            id=fid,
            params=list(params),
            returns=func_returns,
            local_assignments=local_assignments,
            positional_params=list(positional_params),
            keyword_only_params=list(keyword_only_params),
            vararg=vararg_name,
            kwarg=kwarg_name,
            defaults=defaults,
            yields=func_yields,
            return_values=return_values,
        )
        self.module_cg.functions[qualname] = fs
        ## Link method to its class summary (created before class body visit).
        if self._class_stack:
            cn = self._class_stack[-1]
            if cn in self.module_cg.classes:
                if self._func_stack:
                    method_qualname = ".".join(self._func_stack)
                else:
                    method_qualname = node.name
                self.module_cg.classes[cn].methods[method_qualname] = fs
        self._bind_decorated_target(node.name, node.decorator_list)

    ## Visit a FunctionDef node and register it as a local definition.
    #  @param node The FunctionDef AST node.
    def visit_FunctionDef(self, node):
        self._visit_function_def(node)

    ## Visit an AsyncFunctionDef node and register it as a local definition.
    #  @param node The AsyncFunctionDef AST node.
    def visit_AsyncFunctionDef(self, node):
        self._visit_function_def(node)

    ## Visit a lambda as a first-class local call-graph target.
    #
    #  Inline callbacks stored in literal containers need the same bounded
    #  parameter substitution as named functions. The source location makes
    #  the identity stable without inferring a name from a dispatch key.
    #  @param node Lambda expression.
    def visit_Lambda(self, node):
        qualname = self._local_lambda_key(node)
        positional_nodes = (
            list(getattr(node.args, "posonlyargs", []))
            + list(node.args.args))
        positional_params = [arg.arg for arg in positional_nodes]
        keyword_only_params = [
            arg.arg for arg in getattr(node.args, "kwonlyargs", [])
        ]
        params = list(positional_params) + list(keyword_only_params)
        vararg_name = (
            node.args.vararg.arg if node.args.vararg is not None else "")
        kwarg_name = (
            node.args.kwarg.arg if node.args.kwarg is not None else "")
        if vararg_name:
            params.append(vararg_name)
        if kwarg_name:
            params.append(kwarg_name)

        defaults = {}
        default_nodes = list(getattr(node.args, "defaults", []))
        if default_nodes:
            default_params = positional_nodes[-len(default_nodes):]
            for arg, default_node in zip(default_params, default_nodes):
                source = self._default_argument_source(
                    self._call_edge_argument_source(default_node))
                if source is not None:
                    defaults[arg.arg] = source
        for arg, default_node in zip(
                getattr(node.args, "kwonlyargs", []),
                getattr(node.args, "kw_defaults", [])):
            if default_node is None:
                continue
            source = self._default_argument_source(
                self._call_edge_argument_source(default_node))
            if source is not None:
                defaults[arg.arg] = source

        self._class_receiver_stack.append("")
        self.push_scope(SCOPE_FUNCTION, qualname)
        for arg in positional_nodes:
            self._bind_target_name(arg.arg, "local", arg, "parameter")
        for arg in getattr(node.args, "kwonlyargs", []):
            self._bind_target_name(arg.arg, "local", arg, "parameter")
        if node.args.vararg is not None:
            self._bind_target_name(
                vararg_name, "local", node.args.vararg, "parameter")
        if node.args.kwarg is not None:
            self._bind_target_name(
                kwarg_name, "local", node.args.kwarg, "parameter")
        self.function_params[qualname] = list(params)
        self._func_stack.append(qualname.rsplit(".", 1)[-1])
        fid = FunctionId(self.module_name or "", qualname)
        self._caller_stack.append(fid)
        self.visit(node.body)
        result = self.trace_source(node.body) or self.get_base(node.body)
        result = result or UnknownSource("lambda result")
        self._caller_stack.pop()
        self._func_stack.pop()
        self.pop_scope()
        self._class_receiver_stack.pop()

        self.return_sources[qualname] = result
        self.module_cg.functions[qualname] = FunctionSummary(
            id=fid,
            params=list(params),
            returns=result,
            positional_params=list(positional_params),
            keyword_only_params=list(keyword_only_params),
            vararg=vararg_name,
            kwarg=kwarg_name,
            defaults=defaults,
        )

    ## Qualify a name or attribute through its current lexical import binding.
    #  @param root Expression evaluated in the defining scope.
    #  @return Dotted import path, or None for non-import or ambiguous bindings.
    def _definition_import_path(self, root):
        attributes = []
        while isinstance(root, ast.Attribute):
            attributes.append(root.attr)
            root = root.value
        if not isinstance(root, ast.Name):
            return None
        binding = self.current_scope().lookup(root.id, skip_parent_classes=True)
        if (binding is None or binding.binding_kind != 'import'
                or not isinstance(normalize_source(binding.source), str)):
            return None
        imported = self._import_binding_sources.get(self._binding_key(binding))
        if not isinstance(imported, str) or not imported:
            return None
        return '.'.join([imported] + list(reversed(attributes)))

    ## Recognize the narrow Keras registration contract that returns its class.
    #  @param node Class decorator expression.
    #  @return True for a proven registration call with literal configuration.
    def _is_class_preserving_keras_registration(self, node):
        if (not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or self.current_scope().kind != SCOPE_MODULE
                or self.wildcard_modules
                or self._definition_import_path(node.func)
                != 'tensorflow.keras.utils.register_keras_serializable'):
            return False
        binding = self.current_scope().lookup(node.func.id)
        statements = self._module_tree.body
        if not any(isinstance(statement, ast.ImportFrom)
                   and statement.lineno == binding.lineno
                   for statement in statements):
            return False
        # Only accept an unconditional module import. Some unsupported writes
        # (e.g. walrus or augmented assignment) retain an old import binding;
        # reject visible writes rather than inheriting that stale identity.
        for statement in statements:
            if binding.lineno <= statement.lineno < node.lineno:
                for item in ast.walk(statement):
                    if (isinstance(item, ast.Name) and item.id == node.func.id
                            and isinstance(item.ctx, (ast.Store, ast.Del))):
                        return False
        # TensorFlow v2.10.0 generic_utils.register_keras_serializable records
        # the class in registries and returns the same arg. Keep this contract
        # specific to that API; unknown decorators may replace the class.
        if len(node.args) > 2:
            return False
        arguments = dict(zip(('package', 'name'), node.args))
        for keyword in node.keywords:
            if keyword.arg not in ('package', 'name') or keyword.arg in arguments:
                return False
            arguments[keyword.arg] = keyword.value
        for name, value in arguments.items():
            if not isinstance(value, ast.Constant):
                return False
            if not (isinstance(value.value, str)
                    or (name == 'name' and value.value is None)):
                return False
        return True

    ## Capture a simple single base's import path at class definition time.
    #  @param node ClassDef node.
    #  @return Import-backed dotted path, or None for unsupported inheritance.
    def _super_base_import_path(self, node):
        if len(node.bases) != 1 or node.keywords:
            return None
        if any(not self._is_class_preserving_keras_registration(decorator)
               for decorator in node.decorator_list):
            return None
        return self._definition_import_path(node.bases[0])

    ## Visit a ClassDef node and register it with its method and base lists.
    #  @param node The ClassDef AST node.
    def visit_ClassDef(self, node):
        super_base_path = self._super_base_import_path(node)
        self.local.add(node.name)
        self._bind_target_name(node.name, "local", node)
        methods = []
        bases = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
        for base_node in node.bases:
            base_symbol = None
            if isinstance(base_node, ast.Name):
                base_symbol = base_node.id
            elif isinstance(base_node, ast.Attribute):
                base_symbol = self._attribute_name(base_node) or self.get_base(base_node)
            else:
                base_symbol = self.get_base(base_node)
            if base_symbol:
                bases.append(base_symbol)
        self.class_methods[node.name] = methods
        self.class_bases[node.name] = bases
        ## Create ClassSummary BEFORE generic_visit so methods can link to it.
        class_id = FunctionId(self.module_name or "", node.name)
        self.module_cg.classes[node.name] = ClassSummary(
            id=class_id,
            bases=list(bases),
            methods={},
            attrs={},
        )
        self._class_stack.append(node.name)
        self._super_base_path_stack.append((
            super_base_path,
            'tensorflow.keras.utils'
            if super_base_path and node.decorator_list else None,
        ))
        self.push_scope(SCOPE_CLASS, node.name)
        self.generic_visit(node)
        self.pop_scope()
        self._class_stack.pop()
        self._super_base_path_stack.pop()
        ## Populate ClassSummary attrs collected during class body visit.
        class_attrs = {}
        for (cn, attr_name), src in self.instance_attrs.items():
            if cn == node.name:
                class_attrs[attr_name] = src
        self.module_cg.classes[node.name].attrs.update(class_attrs)
        self._bind_decorated_target(node.name, node.decorator_list)

    ## Visit a With node and bind context-variable aliases.
    #  @param node The With AST node.
    def visit_With(self, node):
        for item in node.items:
            source = self.trace_source(item.context_expr)
            if item.optional_vars is not None:
                self._target_to_source(item.optional_vars, source)
        self.generic_visit(node)

    ## Visit an AsyncWith node and bind context-variable aliases.
    #  @param node The AsyncWith AST node.
    def visit_AsyncWith(self, node):
        for item in node.items:
            source = self.trace_source(item.context_expr)
            if item.optional_vars is not None:
                self._target_to_source(item.optional_vars, source)
        self.generic_visit(node)

    ## Seed Python shapes that can flow from one loop iteration to the next.
    #
    #  This is a bounded lexical fixed-point for builtin value shapes only.
    #  Every assignment to a candidate name inside the loop must independently
    #  prove the same PythonShape, and any conflicting preheader binding blocks
    #  the seed. Import-backed owners and unresolved values are never inferred.
    #  @param statements Statements in one loop body.
    #  @param excluded_names Loop-target names that are rebound by iteration.
    def _seed_loop_carried_python_shapes(self, statements,
                                         excluded_names=None):
        evidence = {}
        self._collect_loop_shape_assignments(statements, evidence)
        excluded = set(excluded_names or [])
        for name, observations in evidence.items():
            if name in excluded or not observations:
                continue
            if any(shape is None for shape, _ in observations):
                continue
            first_shape = observations[0][0]
            if any(shape != first_shape for shape, _ in observations[1:]):
                continue
            binding = self.current_scope().lookup(
                name, skip_parent_classes=True)
            if binding is not None:
                if (binding.container_kind != first_shape.kind
                        or binding.container_item_kind
                        != first_shape.item_kind):
                    continue
                continue
            source_node = observations[0][1]
            self._bind_target_name(
                name, first_shape, source_node,
                container_kind=first_shape.kind,
                container_item_kind=first_shape.item_kind)

    ## Collect same-loop assignment shapes without revisiting call nodes.
    #
    #  Nested lexical scopes are excluded. Control-flow bodies remain part of
    #  the loop backedge, but one unknown assignment invalidates convergence.
    #  @param statements Statements to scan.
    #  @param evidence Mutable name -> [(PythonShape or None, node)] mapping.
    def _collect_loop_shape_assignments(self, statements, evidence):
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                evidence.setdefault(statement.name, []).append(
                    (None, statement))
                header_nodes = list(statement.decorator_list)
                if isinstance(statement, (ast.FunctionDef,
                                          ast.AsyncFunctionDef)):
                    header_nodes.extend(statement.args.defaults)
                    header_nodes.extend(
                        item for item in statement.args.kw_defaults
                        if item is not None)
                else:
                    header_nodes.extend(statement.bases)
                    header_nodes.extend(item.value
                                        for item in statement.keywords)
                for header_node in header_nodes:
                    self._collect_loop_named_expr_bindings(
                        header_node, evidence)
                continue
            self._collect_loop_named_expr_bindings(statement, evidence)
            if isinstance(statement, ast.Assign):
                shape = self._expression_python_shape(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        evidence.setdefault(target.id, []).append(
                            (shape, statement))
                    else:
                        for name in self._assignment_target_names(target):
                            evidence.setdefault(name, []).append(
                                (None, statement))
                continue
            if isinstance(statement, ast.AnnAssign):
                if isinstance(statement.target, ast.Name):
                    shape = (
                        self._expression_python_shape(statement.value)
                        if statement.value is not None else None)
                    evidence.setdefault(statement.target.id, []).append(
                        (shape, statement))
                else:
                    for name in self._assignment_target_names(
                            statement.target):
                        evidence.setdefault(name, []).append(
                            (None, statement))
                continue
            if isinstance(statement, ast.AugAssign):
                target = statement.target
                if isinstance(target, ast.Name):
                    evidence.setdefault(target.id, []).append(
                        (None, statement))
                continue

            if isinstance(statement, (ast.For, ast.AsyncFor)):
                for name in self._assignment_target_names(statement.target):
                    evidence.setdefault(name, []).append((None, statement))
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    if item.optional_vars is None:
                        continue
                    for name in self._assignment_target_names(
                            item.optional_vars):
                        evidence.setdefault(name, []).append(
                            (None, statement))
            elif isinstance(statement, ast.Try):
                for handler in statement.handlers:
                    if handler.name:
                        evidence.setdefault(handler.name, []).append(
                            (None, handler))
            elif isinstance(statement, ast.Delete):
                for target in statement.targets:
                    for name in self._assignment_target_names(target):
                        evidence.setdefault(name, []).append(
                            (None, statement))
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    name = alias.asname or alias.name.split(".", 1)[0]
                    evidence.setdefault(name, []).append((None, statement))
            elif isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    if alias.name == "*":
                        continue
                    name = alias.asname or alias.name
                    evidence.setdefault(name, []).append((None, statement))

            child_blocks = []
            if isinstance(statement, ast.If):
                child_blocks.extend((statement.body, statement.orelse))
            elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                child_blocks.extend((statement.body, statement.orelse))
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                child_blocks.append(statement.body)
            elif isinstance(statement, ast.Try):
                child_blocks.extend((statement.body, statement.orelse,
                                     statement.finalbody))
                child_blocks.extend(handler.body
                                    for handler in statement.handlers)
            for child_block in child_blocks:
                self._collect_loop_shape_assignments(child_block, evidence)

    ## Return lexical names bound by an assignment target.
    #  Attribute and subscript writes mutate objects but do not rebind their
    #  receiver names. Tuple, list, and starred targets are traversed.
    #  @param target Assignment-target AST node.
    #  @return Set of rebound lexical names.
    def _assignment_target_names(self, target):
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, ast.Starred):
            return self._assignment_target_names(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            names = set()
            for item in target.elts:
                names.update(self._assignment_target_names(item))
            return names
        return set()

    ## Record assignment expressions as unknown loop-local rebindings.
    #  Named expressions may occur inside tests, call arguments, or container
    #  expressions. Their value shape is deliberately not inferred here.
    #  Nested lexical scopes do not share ordinary target bindings with the
    #  surrounding loop. Comprehension targets are ignored automatically, but
    #  assignment expressions inside comprehensions still bind outside them.
    #  @param node AST subtree to inspect.
    #  @param evidence Mutable loop-shape evidence mapping.
    def _collect_loop_named_expr_bindings(self, node, evidence):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.NamedExpr):
            for name in self._assignment_target_names(node.target):
                evidence.setdefault(name, []).append((None, node))
        for child in ast.iter_child_nodes(node):
            self._collect_loop_named_expr_bindings(child, evidence)

    ## Visit a For node and bind the loop variable to the iterator source.
    #  @param node The For AST node.
    def visit_For(self, node):
        target_names = self._assignment_target_names(node.target)
        self._seed_loop_carried_python_shapes(node.body, target_names)
        yields = self._resolve_iterator_yields(node.iter, node.target)
        if isinstance(node.iter, ast.Name):
            binding = self.current_scope().lookup(
                node.iter.id, skip_parent_classes=True)
            # Deferred bodies cannot snapshot an enclosing mutable mapping.
            if binding is not None and binding.container_kind == "dict":
                key_kind = (self._mapping_facts.key_kind(node.iter)
                            if self.current_scope().bindings.get(
                                node.iter.id) is binding else None)
                if key_kind:
                    yields = [(node.target, PythonShape(key_kind), key_kind)]
                else:
                    yields = [(node.target, UnknownSource("unresolved dictionary key"))]
        if yields is not None:
            for item in yields:
                target_elt, source = item[:2]
                container_kind = item[2] if len(item) > 2 else ""
                container_item_kind = item[3] if len(item) > 3 else ""
                self._target_to_source(
                    target_elt, source, "iteration",
                    container_kind=container_kind,
                    container_item_kind=container_item_kind)
        else:
            source = self._iter_source(node.iter)
            item_kind = ""
            if isinstance(node.iter, ast.Name):
                item_kind = (
                    self._lookup_container_kind(node.iter.id, item=True)
                    or "")
            if not item_kind:
                shape = self._expression_python_shape(node.iter)
                item_kind = shape.item_kind if shape is not None else ""
            item_fields = self._expression_container_item_fields(node.iter)
            if item_fields and not item_kind:
                item_kind = "dict"
            if source is None and item_kind:
                source = PythonShape(item_kind)
            self._target_to_source(
                node.target, source, "iteration",
                container_kind=item_kind,
                container_item_fields=item_fields)
        self._record_iteration_binding(node)
        self.generic_visit(node)

    ## Visit a While node with bounded loop-carried Python shape evidence.
    #  @param node The While AST node.
    def visit_While(self, node):
        self._seed_loop_carried_python_shapes(node.body)
        self.generic_visit(node)

    ## Record a generator-loop fact for bounded cross-file propagation.
    #  @param node For-loop AST node.
    def _record_iteration_binding(self, node):
        if (not isinstance(node.iter, ast.Call)
                or not isinstance(node.target, ast.Name)
                or not self._caller_stack):
            return
        callee_name, _ = self._get_call_parts(node.iter)
        self.module_cg.iteration_bindings.append(IterationBinding(
            caller=self._caller_stack[-1],
            callee_name=callee_name,
            callee_source=self.trace_source(node.iter.func),
            target_names=[node.target.id],
            call_lineno=node.iter.lineno,
            call_col_offset=node.iter.col_offset,
        ))

    ## Resolve per-element ownership for for-loop iterator expressions.
    #
    #  For enumerate(X), zip(A, B) etc., decomposes the target tuple
    #  and binds each element to the appropriate ownership source.
    #  @param iter_node The iterator AST node.
    #  @param target The loop target AST node.
    #  @return List of (target_elt, source) pairs, or None to fall back.
    def _resolve_iterator_yields(self, iter_node, target):
        if not isinstance(iter_node, ast.Call):
            if (isinstance(iter_node, ast.Name)
                    and isinstance(target, (ast.Tuple, ast.List))):
                field_sources = self._lookup_tuple_item_sources(
                    iter_node.id)
                if (field_sources is not None
                        and len(field_sources) == len(target.elts)):
                    return [
                        (field_target, source)
                        for field_target, source in zip(
                            target.elts, field_sources)
                    ]
            if (isinstance(iter_node, ast.Attribute)
                    and isinstance(target, (ast.Tuple, ast.List))):
                field_sources = self._lookup_attribute_tuple_item_sources(
                    iter_node)
                if (field_sources is not None
                        and len(field_sources) == len(target.elts)):
                    return [
                        (field_target, source)
                        for field_target, source in zip(
                            target.elts, field_sources)
                    ]
            return None
        func_name = (
            iter_node.func.id if isinstance(iter_node.func, ast.Name)
            else None)

        # os.walk() and os.fwalk() have a stable stdlib yield contract.
        # Every yielded field is a Python-provided value; binding each tuple
        # position to python also lets nested loops preserve filename-string
        # method ownership without treating os itself as the item owner.
        if isinstance(iter_node.func, ast.Attribute):
            func_top, dotted_name = self._resolve_func_top(iter_node.func)
            if (func_top is None
                    and isinstance(iter_node.func.value, ast.Name)):
                root = iter_node.func.value.id
                candidate_top = self.symbols.get_top(root)
                if root in self.import_aliases:
                    func_top = candidate_top
                    dotted_name = iter_node.func.attr
            expected_arity = 3 if dotted_name == "walk" else 4
            if (func_top == "os"
                    and dotted_name in ("walk", "fwalk")
                    and isinstance(target, (ast.Tuple, ast.List))
                    and len(target.elts) == expected_arity):
                result = [
                    (target.elts[0], "python", "str", ""),
                    (target.elts[1], "python", "list", "str"),
                    (target.elts[2], "python", "list", "str"),
                ]
                if expected_arity == 4:
                    result.append((target.elts[3], "python", "", ""))
                return result

        if func_name in self.import_from_symbols:
            imported = self.import_from_symbols[func_name]
            expected_arity = 3 if imported == "os.walk" else 4
            if (imported in ("os.walk", "os.fwalk")
                    and isinstance(target, (ast.Tuple, ast.List))
                    and len(target.elts) == expected_arity):
                result = [
                    (target.elts[0], "python", "str", ""),
                    (target.elts[1], "python", "list", "str"),
                    (target.elts[2], "python", "list", "str"),
                ]
                if expected_arity == 4:
                    result.append((target.elts[3], "python", "", ""))
                return result

        # Explicit iterator result contracts keep the iterator object's owner
        # separate from the ownership of each yielded element.
        if isinstance(target, ast.Name):
            func_top, resolved_name = self._resolve_func_top(iter_node.func)
            element_owner = _match_iterator_element_owner(
                func_top, resolved_name)
            if element_owner is not None:
                element_shape = _match_iterator_element_shape(
                    func_top, resolved_name)
                element_source = CallResult(
                    InstanceMethod(func_top, resolved_name),
                    call_lineno=iter_node.lineno,
                    call_col_offset=iter_node.col_offset,
                    result_source=element_shape or element_owner,
                )
                if element_shape is not None:
                    return [(
                        target, element_source,
                        element_shape.kind, element_shape.item_kind)]
                return [(target, element_source)]
            traced = normalize_source(self.trace_source(iter_node))
            if isinstance(traced, CallResult):
                result_source = normalize_source(traced.result_source)
                if (isinstance(result_source, DerivedResult)
                        and result_source.kind == "iterator"
                        and result_source.sources):
                    return [(target, result_source.sources[0])]

        if func_name is None:
            return None

        # A local generator's yield summary is source evidence, not a
        # callable-name or library-name heuristic.  Use it before the
        # external iterator contracts so local and cross-file propagation
        # can converge on the same representation.
        if isinstance(target, ast.Name):
            summary = self.module_cg.functions.get(func_name)
            if summary is not None and summary.yields is not None:
                return [(target, summary.yields)]

        ## enumerate(X): for i, x in ... → i=python, x from container elements
        if func_name == "enumerate" and iter_node.args:
            container = iter_node.args[0]
            container_source = self._iterator_container_source(container)
            if isinstance(target, ast.Tuple):
                elts = target.elts
                if len(elts) == 2:
                    item_kind = ""
                    if isinstance(container, ast.Name):
                        item_kind = (
                            self._lookup_container_kind(
                                container.id, item=True) or "")
                    if item_kind:
                        return [(elts[0], "python"),
                                (elts[1], "python", item_kind, "")]
                    return [(elts[0], "python"),
                            (elts[1], ContainerIter(container_source) if container_source else None)]
            return [(target, "python")]

        ## zip(A, B, ...): positional propagation from each input container
        if func_name == "zip" and iter_node.args:
            if isinstance(target, ast.Tuple):
                elts = target.elts
                if len(elts) == len(iter_node.args):
                    result = []
                    for elt, arg in zip(elts, iter_node.args):
                        arg_source = self._iterator_container_source(arg)
                        result.append((elt, ContainerIter(arg_source) if arg_source else None))
                    return result

        return None

    ## Return the container identity for iterator yield resolution.
    #
    #  Preserves AST Name identity for container_items lookup;
    #  falls back to trace_source for complex expressions.
    #  @param node The container AST node.
    #  @return Name string or traced source.
    def _iterator_container_source(self, node):
        if isinstance(node, ast.Name):
            return node.id
        return self.trace_source(node)

    ## Visit an AsyncFor node and bind the loop variable to the iterator source.
    #  @param node The AsyncFor AST node.
    def visit_AsyncFor(self, node):
        target_names = self._assignment_target_names(node.target)
        self._seed_loop_carried_python_shapes(node.body, target_names)
        source = self._iter_source(node.iter)
        self._target_to_source(node.target, source, "iteration")
        self.generic_visit(node)

    ## Record one value yielded by the current generator function.
    #  @param node Yield AST node.
    def visit_Yield(self, node):
        if self._func_stack and self._caller_stack:
            if node.value is None:
                source = UnknownSource("bare yield")
            else:
                source = self._call_edge_argument_source(node.value)
                source = self._source_with_module(source)
                if source is None:
                    source = UnknownSource("unresolved yield")
            qualname = self._caller_stack[-1].qualname
            self.call_graph_yield_sources.setdefault(
                qualname, []).append(source)
        self.generic_visit(node)

    ## Record values forwarded by a yield-from expression.
    #  @param node YieldFrom AST node.
    def visit_YieldFrom(self, node):
        if self._func_stack and self._caller_stack:
            source = None
            if isinstance(node.value, ast.Call):
                callee_name = (
                    node.value.func.id
                    if isinstance(node.value.func, ast.Name) else None)
                if callee_name:
                    summary = self.module_cg.functions.get(callee_name)
                    if summary is not None:
                        source = summary.yields
            if source is None:
                source = self._call_edge_iterable_source(node.value)
            source = self._source_with_module(source)
            if source is None:
                source = UnknownSource("unresolved yield from")
            qualname = self._caller_stack[-1].qualname
            self.call_graph_yield_sources.setdefault(
                qualname, []).append(source)
        self.generic_visit(node)

    ## Visit an If node with lexical branch merging.
    #
    #  Snapshots the current scope before the if, visits each branch
    #  independently, then merges the resulting bindings. Module scope
    #  merges every changed binding. Function scope merges only names
    #  assigned directly by this branch; nested branches merge themselves.
    #  TYPE_CHECKING guards are skipped at all levels.
    #  @param node The If AST node.
    def visit_If(self, node):
        self.visit(node.test)
        receiver_guard = self._resolve_receiver_owner_guard(node.test)

        if self._is_type_checking_guard(node):
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
            return

        if self.current_scope().kind != SCOPE_MODULE:
            self._visit_function_if(node, receiver_guard)
            return

        scope_base = self.current_scope().snapshot()
        symbols_base = self.symbols.snapshot()

        self._visit_guarded_nodes(node.body, receiver_guard, node.test)
        scope_left = self.current_scope().snapshot()

        self.current_scope().restore(scope_base)
        self.symbols.restore(symbols_base)

        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            scope_right = self.current_scope().snapshot()
        else:
            scope_right = scope_base

        merged = merge_snapshots(scope_base, scope_left, scope_right)
        for name, value in list(merged.items()):
            if not isinstance(value, Binding):
                merged[name] = Binding(
                    name=name, source=value,
                    scope_kind=self.current_scope().kind,
                )
        self.current_scope().restore(merged)
        for name, binding in merged.items():
            if isinstance(binding, Binding):
                self.symbols.add(name, binding.source)

    ## Visit a function-level if while merging only direct RHS evidence.
    #
    #  Calls inside each branch retain the established sequential visitor
    #  behavior. After both branches are visited, incompatible direct
    #  assignments are replaced by a function_branch SourceSet.
    #  @param node The function-level If node.
    #  @param receiver_guard Optional receiver narrowing evidence.
    def _visit_function_if(self, node, receiver_guard):
        scope_base = self.current_scope().snapshot()
        left_sources = self._direct_branch_assignment_sources(node.body)
        right_sources = self._direct_branch_assignment_sources(node.orelse)

        self._visit_guarded_nodes(node.body, receiver_guard, node.test)
        names = set(left_sources) | set(right_sources)
        # Restore only the independent mapping facts before the other branch.
        # The established ownership visitor retains its existing merge policy.
        for name in names:
            current = self.current_scope().bindings.get(name)
            if current is not None:
                self.current_scope().bindings[name] = replace(
                    current, mapping_value=getattr(
                        scope_base.get(name), "mapping_value", None))
        for statement in node.orelse:
            self.visit(statement)

        # No mapping join is claimed after a conditional rebind. Captured
        # calls inside each branch still have their branch-local evidence.
        for name in names:
            current = self.current_scope().bindings.get(name)
            if current is not None:
                self.current_scope().bindings[name] = replace(
                    current, mapping_value=None)
        for name in names:
            base_binding = scope_base.get(name)
            left_source = left_sources.get(
                name,
                base_binding.source if isinstance(base_binding, Binding)
                else None)
            right_source = right_sources.get(
                name,
                base_binding.source if isinstance(base_binding, Binding)
                else None)
            if left_source is None or right_source is None:
                continue
            left_binding = Binding(name, left_source)
            right_binding = Binding(name, right_source)
            if not self._branch_sources_require_merge(
                    left_binding, right_binding,
                    allow_local_callable=name.startswith("self.")):
                continue
            current = self.current_scope().lookup(name)
            if current is None:
                continue
            merged_source = make_source_set(
                (left_source, right_source), origin="function_branch")
            self.current_scope().bindings[name] = Binding(
                name=name,
                source=merged_source,
                scope_kind=current.scope_kind,
                lineno=current.lineno,
                col_offset=current.col_offset,
                assignment_index=current.assignment_index,
                version=current.version + 1,
                container_kind=current.container_kind,
                container_item_kind=current.container_item_kind,
                callable_key=current.callable_key,
                binding_kind=current.binding_kind,
            )
            if name.startswith("self.") and self._class_stack:
                self.instance_attrs[
                    (self._class_stack[-1], name)
                ] = merged_source

    ## Collect direct assignment RHS sources from branch statements.
    #
    #  Nested control-flow nodes are intentionally excluded because their
    #  own visitors merge their assignments.
    #  @param statements Statements belonging to the current branch.
    #  @return Mapping of assigned local name to RHS source.
    def _direct_branch_assignment_sources(self, statements):
        sources = {}

        def add_target(target, source):
            if isinstance(target, ast.Name):
                sources[target.id] = source
            elif isinstance(target, ast.Attribute):
                name = self._attribute_name(target)
                if name and name.startswith("self."):
                    sources[name] = source
            elif isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    add_target(element, source)

        for statement in statements:
            if isinstance(statement, ast.Assign):
                if isinstance(statement.value, ast.Subscript):
                    source = UnknownSource("branch_subscript")
                elif isinstance(statement.value, (
                        ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    source = (self._expression_python_shape(statement.value)
                              or UnknownSource("branch_rhs"))
                else:
                    source = (
                        self.trace_source(statement.value)
                        or UnknownSource(
                            "branch_ifexp"
                            if isinstance(statement.value, ast.IfExp)
                            else "branch_rhs"))
                if (isinstance(statement.value, ast.Name)
                        and statement.value.id in self.import_from_symbols):
                    source = self.import_from_symbols[statement.value.id]
                for target in statement.targets:
                    add_target(target, source)
            elif isinstance(statement, ast.AnnAssign):
                if isinstance(statement.value, (
                        ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    source = self._expression_python_shape(statement.value)
                else:
                    source = (
                        self.trace_source(statement.value)
                        if statement.value is not None else "local")
                if source is None:
                    source = UnknownSource("branch_rhs")
                add_target(statement.target, source)
            elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                for alias in statement.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if isinstance(statement, ast.ImportFrom):
                        sources[name] = statement.module or alias.name
                    else:
                        sources[name] = alias.name
        return sources

    ## Decide whether two branch bindings prove different owner identities.
    #
    #  Local function results deliberately share one coarse identity because
    #  their return ownership requires inter-procedural analysis. Explicit
    #  local constructors remain distinct from that coarse bucket.
    #  @param left_binding Binding produced by the true branch.
    #  @param right_binding Binding produced by the false branch.
    #  @param allow_local_callable Whether exact local callable alternatives
    #  are meaningful for this binding. This is limited to instance fields
    #  because ordinary local names already have established flow behavior.
    #  @return True when the bindings must remain a SourceSet.
    def _branch_sources_require_merge(self, left_binding, right_binding,
                                      allow_local_callable=False):
        if not (isinstance(left_binding, Binding)
                and isinstance(right_binding, Binding)):
            return False
        left_identity = self._branch_source_identity(left_binding.source)
        right_identity = self._branch_source_identity(right_binding.source)
        if left_identity == right_identity:
            return False
        left_kind = left_identity[0]
        right_kind = right_identity[0]
        if left_kind == "library" and right_kind == "library":
            return True
        if (allow_local_callable
                and (left_kind == "local_callable"
                     or right_kind == "local_callable")):
            return True
        kinds = {left_kind, right_kind}
        if "PythonShape" in kinds and kinds.issubset({
                "PythonShape", "UnknownSource", "local_result", "result",
                "InstanceMethod", "function_branch"}):
            return True
        # An unresolved assignment branch cannot be discarded merely
        # because the other branch is a function/method result.
        if "UnknownSource" in kinds and kinds.intersection(
                ("local_result", "result", "InstanceMethod")):
            return True
        if "function_branch" in kinds:
            return True
        if "local_class" in kinds and (
                "library" in kinds
                or ("UnknownSource" in kinds
                    and (
                        left_identity == (
                            "UnknownSource", "branch_ifexp")
                        or right_identity == (
                            "UnknownSource", "branch_ifexp")))):
            return True
        return False

    ## Return a coarse, evidence-backed identity for branch comparison.
    #  @param source Source value from a branch binding.
    #  @return Hashable identity tuple.
    def _branch_source_identity(self, source):
        source = normalize_source(source)
        if isinstance(source, SourceSet):
            return (
                source.origin or "set",
                tuple(sorted(
                    repr(self._branch_source_identity(item))
                    for item in source.sources)),
            )
        if isinstance(source, CallResult):
            result_source = normalize_source(source.result_source)
            if isinstance(result_source, str) and result_source:
                return ("result", result_source.split(".")[0])
            callee = source.callee
            if isinstance(callee, str):
                first = callee.split(".")[0]
                imported = self._imported_top_for_branch(first)
                if imported:
                    return ("library", imported)
                if callee in self.class_methods or first in self.class_methods:
                    return ("local_class", callee)
            return ("local_result",)
        if isinstance(source, str):
            if source in self.import_from_symbols.values():
                return ("local_callable", source)
            first = source.split(".")[0]
            imported = self._imported_top_for_branch(first)
            if imported:
                return ("library", imported)
            if source in self.class_methods or first in self.class_methods:
                return ("local_class", source)
            return (source if source in ("python", "unknown") else "local",)
        return (type(source).__name__, source_display(source))

    ## Resolve a single-file import name to its top-level library.
    #  @param name Root symbol used by a branch source.
    #  @return Top-level import name or an empty string.
    def _imported_top_for_branch(self, name):
        imported = self.import_from_symbols.get(name)
        if imported:
            return imported.split(".")[0]
        if name in self.import_aliases:
            direct = self.symbols.direct.get(name, name)
            if isinstance(direct, str):
                return direct.split(".")[0]
            return name.split(".")[0]
        return ""

    ## Visit a conditional expression with true-branch receiver narrowing.
    #
    #  @param node The IfExp AST node.
    def visit_IfExp(self, node):
        self.visit(node.test)
        receiver_guard = self._resolve_receiver_owner_guard(node.test)
        self._visit_guarded_nodes((node.body,), receiver_guard)
        self.visit(node.orelse)

    ## Check whether an If node guards on TYPE_CHECKING.
    #  @param node The If AST node.
    #  @return True if the test is a bare TYPE_CHECKING reference.
    def _is_type_checking_guard(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return True
        if isinstance(node.test, ast.Attribute):
            if node.test.attr == "TYPE_CHECKING":
                return True
        return False

    ## Visit a Try node with conservative lexical branch merging.
    #
    #  At module level, each except handler and the else clause are treated
    #  as independent branches merged conservatively.  At function level,
    #  falls back to generic_visit (deferred to Phase 6 CFG).
    #  @param node The Try AST node.
    def visit_Try(self, node):
        if self.current_scope().kind != SCOPE_MODULE:
            self.generic_visit(node)
            return

        scope_base = self.current_scope().snapshot()
        symbols_base = self.symbols.snapshot()

        for stmt in node.body:
            self.visit(stmt)
        scope_try = self.current_scope().snapshot()
        symbols_try = self.symbols.snapshot()

        all_branches = [scope_try]
        for handler in node.handlers:
            self.current_scope().restore(scope_base)
            self.symbols.restore(symbols_base)
            if handler.type:
                self.visit(handler.type)
            if handler.name:
                self._bind_target_name(handler.name, "local", handler, "variable")
            for stmt in handler.body:
                self.visit(stmt)
            all_branches.append(self.current_scope().snapshot())

        if node.orelse:
            self.current_scope().restore(scope_try)
            self.symbols.restore(symbols_try)
            for stmt in node.orelse:
                self.visit(stmt)
            all_branches.append(self.current_scope().snapshot())

        merged = scope_base
        for branch in all_branches:
            merged = merge_snapshots(scope_base, merged, branch)
        for name, value in list(merged.items()):
            if not isinstance(value, Binding):
                merged[name] = Binding(
                    name=name, source=value,
                    scope_kind=self.current_scope().kind,
                )
        self.current_scope().restore(merged)

        for name, binding in merged.items():
            if isinstance(binding, Binding):
                self.symbols.add(name, binding.source)

        for stmt in node.finalbody:
            self.visit(stmt)

    ## Common handler for all comprehension node types.
    #  @param node A ListComp, SetComp, DictComp, or GeneratorExp AST node.
    def _visit_comprehension(self, node):
        self.push_scope(SCOPE_COMPREHENSION, "<comprehension>")
        for gen in node.generators:
            target_names = [
                item.id for item in ast.walk(gen.target)
                if isinstance(item, ast.Name)
            ]
            self.comprehension_targets.update(target_names)
            source = self._iter_source(gen.iter)
            self._target_to_source(gen.target, source)
        self.generic_visit(node)
        self.pop_scope()

    ## Visit a ListComp node and bind loop variables to the iterator source.
    #  @param node The ListComp AST node.
    def visit_ListComp(self, node):
        self._visit_comprehension(node)

    ## Visit a DictComp node and bind loop variables to the iterator source.
    #  @param node The DictComp AST node.
    def visit_DictComp(self, node):
        self._visit_comprehension(node)

    ## Visit a SetComp node and bind loop variables to the iterator source.
    #  @param node The SetComp AST node.
    def visit_SetComp(self, node):
        self._visit_comprehension(node)

    ## Visit a GeneratorExp node and bind loop variables to the iterator source.
    #  @param node The GeneratorExp AST node.
    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node)

    ## Visit a Global node and mark names for module-scope routing.
    #  @param node The Global AST node.
    def visit_Global(self, node):
        for name in node.names:
            self._global_names.add(name)
        self.generic_visit(node)

    ## Visit a Nonlocal node. First edition: no-crash only.
    #  @param node The Nonlocal AST node.
    def visit_Nonlocal(self, node):
        self.generic_visit(node)

    ## Build a positional source summary for a tuple return expression.
    #
    #  The tuple itself has no single owner. Preserve each element source so
    #  a cross-file call edge can bind ``left, right = make_pair()`` by
    #  position without promoting one element's owner to the whole tuple.
    #  @param value Tuple AST node.
    #  @return DerivedResult with one source per tuple element, or None when
    #  any element cannot be traced.
    def _tuple_return_source(self, value):
        if not isinstance(value, ast.Tuple):
            return None
        elements = []
        for element in value.elts:
            source = self.trace_source(element)
            if source is None:
                return None
            elements.append(normalize_source(source))
        return DerivedResult("tuple", tuple(elements))

    ## Check whether a statement sequence definitely cannot fall through.
    #  @param statements Function or branch statements.
    #  @return True for a proven return/raise on every path.
    def _block_exits(self, statements):
        for statement in statements:
            if isinstance(statement, (ast.Return, ast.Raise)):
                return True
            if (isinstance(statement, ast.If)
                    and self._block_exits(statement.body)
                    and self._block_exits(statement.orelse)):
                return True
        return False

    ## Preserve arithmetic return operands at their evaluation point.
    #  @param node Return expression or one of its operands.
    #  @return Structured value source, including unsupported operands.
    def _return_expression_source(self, node):
        if isinstance(node, ast.BinOp):
            return DerivedResult("expression", (
                self._return_expression_source(node.left),
                self._return_expression_source(node.right)),
                type(node.op).__name__)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return PythonShape("bool")
            return DerivedResult("expression", (
                self._return_expression_source(node.operand),),
                type(node.op).__name__)
        if isinstance(node, ast.Name):
            binding = self.current_scope().lookup(
                node.id, skip_parent_classes=True)
            if binding is None:
                return UnknownSource("unbound return operand")
        shape = self._expression_python_shape(node)
        if shape is not None:
            if shape.kind == "NoneType":
                return UnknownSource("non-arithmetic None operand")
            return shape
        dependency = self._parameter_dependency_source(
            node, expression_context=True)
        return (dependency or self._call_edge_argument_source(node)
                or UnknownSource("unresolved return operand"))

    ## Visit a Return node and record return-value flow for the function.
    #  @param node The Return AST node.
    def visit_Return(self, node):
        if node.value is not None:
            self._mapping_facts.escape(node.value)
        if self._caller_stack:
            return_key = self._caller_stack[-1].qualname
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                elements = [
                    self._expression_python_shape(element)
                    or self._call_edge_argument_source(element)
                    or UnknownSource("unresolved returned element")
                    for element in node.value.elts
                ]
            else:
                returned = None
                if node.value is not None:
                    returned = (self._expression_python_shape(node.value)
                                or self._call_edge_argument_source(node.value))
                elements = [ContainerIter(returned) if returned is not None
                            else UnknownSource("unresolved returned iterable")]
            self.return_element_sources.setdefault(return_key, []).extend(
                elements)
        cg_source = None
        if self._func_stack and node.value is not None:
            func_name = self._func_stack[-1]
            source = None
            if isinstance(node.value, (ast.BinOp, ast.UnaryOp)):
                source = self._return_expression_source(node.value)
            elif (isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)):
                source = self._parameter_dependency_source(
                    node.value, expression_context=True)
            tuple_source = self._tuple_return_source(node.value)
            result_kind = _container_kind(node.value)
            # A tuple owns only the aggregate object.  Its unpacked items may
            # have unrelated owners, so do not promote the function's entire
            # return contract to Python from a tuple literal alone.
            if tuple_source is not None:
                source = tuple_source
            elif source is not None:
                pass
            elif result_kind is not None and result_kind != "tuple":
                source = "python"
            elif (isinstance(node.value, ast.Name)
                  and self._lookup_container_kind(node.value.id) is not None):
                source = "python"
            else:
                source = self.trace_source(node.value)
            if source:
                if isinstance(source, str) and source in self.symbols.direct:
                    s = self.symbols.direct[source]
                    new_src = s if s else source
                else:
                    new_src = source
                if (source == "local" and isinstance(node.value, ast.Name)
                        and node.value.id in self.function_params.get(func_name, [])):
                    new_src = node.value.id
                ## Write qualified key for class methods; bare key only
                ## for non-class functions to prevent cross-class pollution.
                if self._class_stack:
                    qkey = self._class_stack[-1] + "." + func_name
                    return_key = qkey
                else:
                    return_key = func_name
                cg_source = tuple_source or new_src
                if tuple_source is not None:
                    old_cg = self.call_graph_return_sources.get(return_key)
                    self.call_graph_return_sources[return_key] = (
                        make_source_set(
                            [old_cg, new_src] if old_cg else [new_src],
                            origin="return"))
                else:
                    old_legacy = self.return_sources.get(return_key)
                    self.return_sources[return_key] = make_source_set(
                        [old_legacy, new_src]
                        if old_legacy else [new_src],
                        origin="return")
                    # Preserve a mixed tuple/non-tuple contract when a
                    # function has both return shapes across branches.
                    if return_key in self.call_graph_return_sources:
                        old_cg = self.call_graph_return_sources[return_key]
                        self.call_graph_return_sources[return_key] = (
                            make_source_set(
                                [old_cg, new_src], origin="return"))
                self._add_symbol_ref(
                    func_name + ".return", source, "return", node)
        if self._caller_stack:
            # Protocol resolution needs every return alternative, including
            # scalar/None branches that have no legacy symbol provenance.
            if node.value is None:
                cg_source = PythonShape("NoneType")
            else:
                shape = self._call_edge_protocol_source(node.value)
                if isinstance(node.value, ast.Tuple):
                    cg_source = DerivedResult("tuple", tuple(
                        self._call_edge_protocol_source(element)
                        or self._call_edge_argument_source(element)
                        or UnknownSource("unresolved returned tuple item")
                        for element in node.value.elts))
                elif shape is not None:
                    cg_source = shape
                elif isinstance(node.value, (ast.Name, ast.Subscript)):
                    cg_source = self._call_edge_argument_source(node.value)
                elif cg_source is None:
                    cg_source = shape
            cg_source = cg_source or UnknownSource("unresolved return branch")
            return_key = self._caller_stack[-1].qualname
            old_cg = self.call_graph_return_values.get(return_key)
            self.call_graph_return_values[return_key] = make_source_set(
                [old_cg, cg_source] if old_cg else [cg_source], origin="return")
        self.generic_visit(node)


## Analyze a single Python source string and return structured results.
#
#  Convenience function that parses source code and runs a full analysis
#  pass, returning a FileAnalysis object.
#  @param source Python source code as a string.
#  @param file_path Optional file path for the FileAnalysis record.
#  @return FileAnalysis with symbols, chains, and API calls.
## Analyze a single source string and return per-file results.
#  @param source Python source code string.
#  @param file_path Optional file path for reporting.
#  @return FileAnalysis object.
def analyze_source(source, file_path="<string>"):
    tree = ast.parse(source)
    tracer = SingleFileAnalyzer(file_path=file_path)
    tracer.visit(tree)
    return FileAnalysis(
        file_path=file_path,
        module_name="",
        symbols=dict(tracer.symbols.direct),
        chains=dict(tracer.symbols.chains),
        api_calls=[
            ApiCall(
                expression=c['api'],
                top_library=c['top'],
                base_symbol=source_display(c.get('base', '')),
                chain=c.get('chain', []),
                file_path=file_path,
                lineno=c.get('lineno', 0),
                col_offset=c.get('col_offset', 0),
                end_lineno=c.get('end_lineno', 0),
                end_col_offset=c.get('end_col_offset', 0),
                func_name=c.get('func_name', ''),
                parameters=c.get('parameters', ''),
                resolved_func=c.get('func_name', ''),
                resolved_chain=[c.get('func_name', ''), c.get('func_name', ''), c.get('top', '')],
            )
            for c in tracer.api_calls
        ],
    )
