"""Static analysis of Python training scripts via AST.

Goal: flag *suspected* preprocessing leakage patterns such as:

- fitting a scalar/imputer/encoder on the whole dataset before any split
- oversampling before train/test split
- feature selection fit on all observations
- TRANSFORMING test data with a transformer fitted on all data

These are static suspicions only — they never prove runtime leakage. They become
findings only when combined with experimental evidence.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class CodeFinding:
    line: int
    pattern: str
    message: str
    severity: str = "medium"  # low | medium | high
    evidence_status: str = "suspicious"
    code_snippet: str = ""


# Transformers/techniques whose fit-before-split is suspicious.
FIT_NAMES = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "QuantileTransformer", "PowerTransformer",
    "SimpleImputer", "KNNImputer", "IterativeImputer",
    "OneHotEncoder", "OrdinalEncoder", "PolynomialFeatures",
    "PCA", "SelectKBest", "SelectFromModel", "RFE", "VarianceThreshold",
    "RandomOverSampler", "SMOTE", "ADASYN", "TomekLinks",
}
TRANSFORM_METHODS = {"fit", "fit_transform", "transform"}
SPLIT_NAMES = {"train_test_split", "StratifiedKFold", "KFold", "cross_val_score", "GridSearchCV"}


@dataclass
class ASTAnalysis:
    findings: list[CodeFinding] = field(default_factory=list)
    used_ml_libraries: list[str] = field(default_factory=list)
    saw_split: bool = False
    parse_error: Optional[str] = None

    @property
    def high_risk_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")


_EMPTY_FIT_MARKER: Callable[[], CodeFinding] = lambda: None  # type: ignore[assignment]


class _LeakVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[CodeFinding] = []
        self.saw_split = False
        self._split_outputs: set[str] = set()
        # objects we have applied a suspicious fit to (name -> (line, snippet))
        self._fitted_objects: dict[str, tuple[int, str]] = {}
        self._library_imports: set[str] = set()

    def _snippet(self, node: ast.AST) -> str:
        try:
            return ast.get_source_segment(self._source, node) or ""
        except Exception:  # noqa: BLE001
            return ""

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._library_imports.add(node.module.split(".")[0])
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._library_imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def _name_of(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._name_of(node.value)
        return None

    def _is_transformer_call(self, node: ast.Call) -> bool:
        fname = self._name_of(node.func)
        return bool(fname and fname in FIT_NAMES)

    def _is_split_call(self, node: ast.Call) -> bool:
        fname = self._name_of(node.func)
        return bool(fname and fname in SPLIT_NAMES)

    def visit_Assign(self, node: ast.Assign) -> None:
        # detect split output variables
        if isinstance(node.value, ast.Call) and self._is_split_call(node.value):
            self.saw_split = True
            for t in node.targets:
                if isinstance(t, ast.Tuple):
                    for elt in t.elts:
                        nm = self._name_of(elt)
                        if nm:
                            self._split_outputs.add(nm)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value and isinstance(node.value, ast.Call) and self._is_split_call(node.value):
            self.saw_split = True
            nm = self._name_of(node.target)
            if nm:
                self._split_outputs.add(nm)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fname = self._name_of(node.func)
        if fname in TRANSFORM_METHODS and isinstance(node.func, ast.Attribute):
            obj = self._name_of(node.func.value)
            # inline constructor pattern: StandardScaler().fit_transform(X)
            if isinstance(node.func.value, ast.Call):
                ctor = self._name_of(node.func.value.func)
                if ctor in FIT_NAMES:
                    self._record_inline_fit(node)
                    return
            if obj and obj in FIT_NAMES:
                return  # constructor inside assignment, skip
            # object.fit(...) / object.fit_transform(...)
            if fname in ("fit", "fit_transform") and obj:
                self._fitted_objects[obj] = (node.lineno, self._snippet(node))
            return

        if self._is_transformer_call(node):
            # check if fitted inline on a non-train variable
            self._record_inline_fit(node)
            return

        self.generic_visit(node)

    def _record_inline_fit(self, node: ast.Call) -> None:
        # e.g. OneHotEncoder().fit_transform(X) or StandardScaler().fit_transform(X)
        if not isinstance(node.func, ast.Attribute):
            return
        meth = node.func.attr
        if meth not in ("fit", "fit_transform"):
            return
        args = node.args or []
        if not args:
            return
        data_arg = self._name_of(args[0])
        if data_arg is None:
            return
        if data_arg in self._split_outputs:
            return  # fit on train split is fine
        # data not obviously one of the split outputs
        self.findings.append(
            CodeFinding(
                line=node.lineno,
                pattern="preprocessing_before_split",
                severity="high",
                message=(
                    f"Transformer method `{meth}` called on `{data_arg}` — this argument "
                    "does not appear to be a train-split output, so the transform may be "
                    "fitted on the whole dataset (suspected leakage)."
                ),
                code_snippet=self._snippet(node),
            )
        )

    def visit_Expr(self, node: ast.Expr) -> None:
        self.generic_visit(node)


def analyze_script(source: str, filename: str = "<training_script>") -> ASTAnalysis:
    analysis = ASTAnalysis()
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        analysis.parse_error = f"Unable to parse script: {exc.msg} at line {exc.lineno}"
        return analysis

    visitor = _LeakVisitor()
    visitor._source = source  # type: ignore[attr-defined]
    visitor.visit(tree)

    analysis.findings = visitor.findings
    analysis.saw_split = visitor.saw_split

    for imp in sorted(visitor._library_imports):
        if imp.lower() in {"sklearn", "sklearn", "imblearn", "pandas", "numpy", "xgboost", "lightgbm"}:
            analysis.used_ml_libraries.append(imp)

    # global sanity notes
    if not analysis.saw_split:
        analysis.findings.append(
            CodeFinding(
                line=1,
                pattern="missing_split",
                severity="low",
                message="No train/test split detected in the script; evaluation may be misleading.",
                evidence_status="suspicious",
            )
        )
    return analysis
