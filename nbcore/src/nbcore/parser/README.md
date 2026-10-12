# NBFix's own parser + IR

Replaces the analyzer's previous external dependencies -- `gast`/stdlib
`ast` (parsing), `beniget` (def-use chains), and `externals/simple_cfg`
(CFG construction) -- with a self-contained pipeline built on
[Lark](https://github.com/lark-parser/lark).

## Pipeline

```
source text
  -> lark_parser.py      (grammar/notebook_python.lark + Indenter postlex)
  -> ast_transformer.py  -> ast_nodes.py        (our own AST, P0 constructs)
  -> cfg_builder.py       -> cfg_nodes.py        (graph CFG, same node
                                                   shapes analyses already
                                                   expect: Node/
                                                   AssignmentNode/CondNode/
                                                   BBorBInode/...)
  -> def_use.py                                  (unbound-names + per-cell
                                                   assign/import/func
                                                   bookkeeping)
```

`IR/intermediate_representations.py` wires these into the same
`.AST`/`.CFG`/`.UDA` surface the four analyses (dataleak / stale / idle /
isolated) already consume, so none of their own logic needed to change --
only their imports moved from `simple_cfg`/`ast`/`gast` to this package.

## Implemented (grammar tiers P0, most of P1)

Assignments (plain/chained/tuple-unpack/augmented/annotated), attribute and
subscript access (including slicing), calls (positional/keyword/`*args`/
`**kwargs`), all binary/boolean/comparison/unary operators, `if`/`elif`/
`else`, `while`/`for` (with `else`), `try`/`except`/`else`/`finally`,
`with`, `import`/`from...import` (including relative), function and class
definitions (including decorators), `return`/`raise`/`break`/`continue`/
`pass`/`global`, list/tuple/dict literals, f-strings (see gap below).

**Comprehensions** (list/set/dict, and generator expressions -
`(x for x in y)`, `sum(x for x in y)`) are implemented
(`ast_nodes.comprehension`/`ListComp`/`SetComp`/`DictComp`/`GeneratorExp`,
built via `comp_for`/`comp_if`/`set_comp`/`dict_comp`/`genexpr_arg` in
`ast_transformer.py`), for a **single `for` clause only** - the grammar's
`comp_for` rule has no recursive second `comp_for`, so chained
comprehensions (`[x for x in a for y in b]`) aren't expressible yet; that
needs a grammar change, tracked as a separate follow-up, not the same gap.
No new CFG node types were needed for this (see `cfg_builder.py`'s
`_collect_names` for the one related fix: comprehension-bound names are
now excluded from "used names," which matters once a Store-context name
can appear nested inside the same expression subtree as free variables -
previously never possible for any supported construct).

## Same-cell function-call inlining: reduced scope, TODO to broaden

Implemented for a **reduced case only** (`CFGBuilder._inline_call` /
`_check_inline_eligible` in `cfg_builder.py`): a call to a locally-defined
function is inlined -- spliced into the caller's CFG with its parameters
and locals renamed uniquely per call site so they can't collide with the
caller's variables or with another call to the same function -- only when
*all* of the following hold:

- the function has plain positional parameters only: no defaults, no
  `*args`/`**kwargs`, no keyword-only params;
- the call passes plain positional arguments only, matching the parameter
  count exactly: no keywords, no `*`/`**` unpacking at the call site;
- the function's body contains no call to another locally-defined function
  (this is what rules out both recursion and nested inlining -- a body
  that calls itself, directly or via another local helper, is rejected);
- the result, if used, comes from a single `return <expr>` that is the
  function's very last statement (no early/multiple returns); if the
  result is discarded (a bare `f(...)` statement) the function may have no
  `return` at all.
- the call's result isn't tuple/multi-target-unpacked (`a, b = f(...)`).

**Any cell that calls a locally-defined function outside this envelope
fails to build its IR at all** -- `_check_inline_eligible` raises
`NotImplementedError` naming exactly which condition failed, the same
"reject loudly, don't silently mis-analyze" policy used everywhere else in
this package. **Broadening this (defaults/kwargs/`*args`/`**kwargs`,
multiple/early returns, nested calls to other local functions, recursion)
is a tracked TODO, not considered done.** Every call that doesn't resolve
to a locally-defined function at all (the overwhelming majority of
notebook code -- pandas/numpy/sklearn calls) is unaffected and still
treated as black-box (`BBorBInode`), as before.

## Deferred (tracked, not dropped)

Backlog, in the order we're planning to work through it (most
notebook-real-world-impact first). Verified directly against
`ast_transformer.py` + `notebook_python.lark`, not just this doc's own
prior summary -- two items (bytes literals, bare set literals) were
missing from here until this pass.

Hard failures -- raise `NotImplementedError` naming the construct
(`ast_transformer.py`'s `__default__`), not a silent misparse:

- [x] ~~Comprehensions~~ -- done, see Implemented above.
- [ ] `lambda` -- flagged P1 in the grammar, "very common in `.apply()`".
  **Next priority** now that comprehensions are done.
- [ ] Ternary expressions (`x if y else z`) -- P1.
- [ ] Bare set literals (`{1, 2, 3}`) -- distinct from set *comprehensions*
  above; fails even with no `for` involved (`dict_or_set_maker`'s
  `set_literal` alias has no transformer method).
- [ ] Walrus operator (`:=`) -- P2.
- [ ] `assert` -- P1.
- [ ] `del` -- P1.
- [ ] `nonlocal` -- P2.
- [ ] `async`/`await` (`async def`/`async for`/`async with`) -- P2.
- [ ] Dict unpacking in a literal (`{**a, **b}`) -- P2.

Parse-level failures -- the grammar has no rule at all yet, so these are a
raw Lark parse exception, not today's clean `NotImplementedError`; need a
grammar change before a transformer method makes sense:

- [ ] PEP 570 positional-only params (`def f(x, /, y): ...`).
- [ ] `match`/`case` structural pattern matching (PEP 634).
- [ ] `type` alias statements (PEP 695).
- [ ] Exception groups (`except*`).

Silent semantic gaps -- parses successfully, produces an approximated
result, **no error at all**. Arguably higher-risk than the hard failures
above precisely because nothing signals the analysis is working off wrong
data:

- [ ] **f-strings are opaque literals**: `f"{x}"` round-trips as the
  literal text `"{x}"` -- the interpolated expression is never parsed, so a
  name used only inside an f-string won't be picked up by def-use analysis.
  The old pipeline did walk into these (gast's `JoinedStr`/`FormattedValue`).
- [ ] **Single-element tuple with trailing comma silently loses its
  tuple-ness**: `f = (1,)` parses as `Constant(value=1)`, i.e. plain `1`,
  not `Tuple(elts=[1])` -- found incidentally while implementing
  comprehensions (`testlist_comp_tuple`'s `len(children) == 1` branch can't
  distinguish "one parenthesized expression" from "one element + trailing
  comma," since Lark's `maybe_placeholders` swallows the literal `","`
  without leaving a trace). Not a comprehension bug and not touched by
  that work - flagged here as newly discovered, unrelated, still open.
- [ ] **Byte-string literals are kept as `str`, not `bytes`**
  (`_parse_string_literal` in `ast_transformer.py`).
- [ ] Decorator call-arguments aren't preserved (the decorator
  name/attribute chain is; e.g. `@app.route("/x")`'s `"/x"` argument is
  dropped) since decorators aren't consumed by the CFG builder yet.
- [ ] The `MUTATORS` rewrite (`lst.append(x)` treated like
  `lst += lst.append(x)` so mutation propagates taint onto the receiver)
  isn't implemented.

## Known simplifications (not gaps, but worth knowing about)

- `def_use.py`'s `DefUseChains` computes a cell's inputs as the names
  live at the entry of its CFG (`liveness.py`), so they respect statement
  order, branches, loops and inlined calls: `x = f(x)` reads an earlier
  `x`, and a name bound on only one branch is still an input. What the CFG
  doesn't carry is handled cell-wide instead: imports (no CFG node) bind
  everywhere, `global` assignments in a function count as cell bindings,
  and free variables of def/class bodies (opaque single nodes) are inputs
  unless the cell binds them somewhere. `except E as e` is read/bound at
  the TryNode, and comprehensions are flat.
- `try`/`except` control flow connects the `TryNode` itself to each
  handler (approximating "the body might raise") rather than simple_cfg's
  finer "any statement in the body can raise" edges. An
  under-approximation, not an over-approximation.
- Tuple-unpacking a call's return value (`a, b = f(...)`) creates one
  black-box-call node *per target*, matching simple_cfg's own (somewhat
  redundant) behavior for that pattern rather than a single shared call
  node.

## Tests

`framework/tests/test_parser_ast.py`, `test_cfg_builder.py`,
`test_cfg_inlining.py`, `test_def_use.py` exercise this package directly.
The full framework suite
(`framework/tests/test_*.py`) exercises it end-to-end through the four
analyses and passes in full -- see git history for the baseline comparison
against the old gast/beniget/simple_cfg pipeline.
