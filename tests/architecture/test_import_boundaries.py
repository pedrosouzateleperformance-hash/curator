from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _python_files(package: str) -> list[Path]:
    folder = SRC_ROOT / package
    return [path for path in folder.rglob("*.py") if "__pycache__" not in path.parts]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    return imported


def test_domain_imports_no_application_or_adapters() -> None:
    disallowed = ("src.application", "src.adapters", "src.pipeline")
    for file in _python_files("domain"):
        imports = _imports(file)
        assert not any(imp.startswith(disallowed) for imp in imports), f"{file} imports forbidden layer: {imports}"


def test_ports_are_interface_only() -> None:
    for file in _python_files("ports"):
        tree = ast.parse(file.read_text(), filename=str(file))
        has_protocol = any(
            isinstance(node, ast.ClassDef)
            and any(
                (isinstance(base, ast.Name) and base.id in {"Protocol", "ABC"})
                or (isinstance(base, ast.Attribute) and base.attr in {"Protocol", "ABC"})
                for base in node.bases
            )
            for node in tree.body
        )
        assert has_protocol or file.name == "__init__.py", f"Port module must define protocol/ABC interfaces only: {file}"


def test_adapters_depend_on_ports() -> None:
    for file in _python_files("adapters"):
        if file.name == "__init__.py":
            continue
        imports = _imports(file)
        assert any(imp.startswith("src.ports") for imp in imports), f"Adapter missing dependency on ports: {file}"


def test_pipeline_compose_use_cases_only() -> None:
    for file in _python_files("pipeline"):
        if file.name != "orchestrator.py":
            continue
        imports = _imports(file)
        assert any(imp.startswith("src.application.use_cases") for imp in imports), "Pipeline orchestrator must compose application use cases"
        assert not any(imp.startswith("src.adapters.") and "__init__" not in imp for imp in imports), (
            "Pipeline orchestrator should rely on dependency wiring and use-cases, not adapter internals"
        )
