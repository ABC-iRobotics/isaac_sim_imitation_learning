import importlib
import inspect
import pkgutil
import types
from typing import Any, Dict, Iterable, Optional, Tuple


def _try_import(package: str):
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError:
        return None


def _find_local_commands_package(
    host: Any,
    local_subpackage: str = "commands",
) -> Optional[str]:
    """
    Robust local commands package discovery that works even when host.__class__.__module__ == "__main__".

    Strategy:
      1) Resolve a filesystem path for the host class (or its module).
      2) Walk upward from that directory and look for "<dir>/<local_subpackage>/__init__.py".
      3) If found, convert that directory to a dotted import path by matching a sys.path root.

    Debug output is printed to stdout.
    """
    import inspect
    import sys
    from pathlib import Path

    def dbg(msg: str) -> None:
        print(f"[registry][debug] {msg}")

    cls = host.__class__
    dbg(f"Host class: {cls.__name__}, __module__={cls.__module__!r}")

    # --- 1) Find a concrete source file path for the class/module
    candidate_files = []

    # a) Prefer inspect.getfile(cls) when possible (works often even for
    # __main__)
    try:
        f = inspect.getfile(cls)
        if f:
            candidate_files.append(f)
            dbg(f"inspect.getfile(cls) -> {f}")
    except Exception as e:
        dbg(f"inspect.getfile(cls) failed: {type(e).__name__}: {e}")

    # b) Fallback to module __file__
    try:
        mod = sys.modules.get(cls.__module__)
        mod_file = getattr(mod, "__file__", None) if mod else None
        if mod_file:
            candidate_files.append(mod_file)
            dbg(f"sys.modules[{cls.__module__!r}].__file__ -> {mod_file}")
        else:
            dbg(f"sys.modules[{cls.__module__!r}].__file__ not available")
    except Exception as e:
        dbg(f"module __file__ lookup failed: {type(e).__name__}: {e}")

    # c) If still nothing, try __main__.__file__ as last resort
    if not candidate_files:
        main_mod = sys.modules.get("__main__")
        main_file = getattr(main_mod, "__file__", None) if main_mod else None
        if main_file:
            candidate_files.append(main_file)
            dbg(f"fallback __main__.__file__ -> {main_file}")

    if not candidate_files:
        dbg("No candidate file path found; cannot determine local commands package.")
        return None

    # Use the first viable path
    src_file = Path(candidate_files[0]).resolve()
    if not src_file.exists():
        dbg(f"Resolved source file does not exist: {src_file}")
        return None

    start_dir = src_file.parent
    dbg(f"Starting directory for upward search: {start_dir}")

    # --- 2) Walk upward and look for "<dir>/commands/__init__.py"
    found_commands_dir: Optional[Path] = None
    for depth, current in enumerate([start_dir, *start_dir.parents]):
        commands_dir = current / local_subpackage
        init_py = commands_dir / "__init__.py"
        dbg(f"Search depth={depth}: checking {init_py}")

        if init_py.exists() and init_py.is_file():
            found_commands_dir = commands_dir
            dbg(f"FOUND commands package dir: {found_commands_dir}")
            break

    if not found_commands_dir:
        dbg(f"No '{local_subpackage}/__init__.py' found while walking upward.")
        return None

    # --- 3) Convert filesystem path to dotted import path by matching a sys.path entry
    # We need the package path relative to a sys.path root, like:
    #   sys.path root: /.../project_root
    #   found_commands_dir: /.../project_root/my_pkg/commands
    #   dotted: "my_pkg.commands"
    sys_path_roots = []
    for p in sys.path:
        if not p:
            continue
        try:
            sys_path_roots.append(Path(p).resolve())
        except Exception:
            continue

    dbg(f"sys.path roots (resolved): {len(sys_path_roots)} entries")

    for root in sys_path_roots:
        try:
            rel = found_commands_dir.relative_to(root)
        except Exception:
            continue

        # rel: e.g. my_pkg/commands  -> "my_pkg.commands"
        parts = list(rel.parts)
        if not parts:
            continue

        dotted = ".".join(parts)
        dbg(f"Matched sys.path root: {root}")
        dbg(f"Relative path: {rel} -> dotted package: {dotted}")

        # Sanity check: try importing it
        try:
            _try_import(dotted)
            dbg(f"Import test OK for {dotted}")
            return dotted
        except Exception as e:
            dbg(f"Import test FAILED for {dotted}: {type(e).__name__}: {e}")
            # keep trying other roots

    dbg("Could not map commands directory to an importable dotted package via sys.path.")
    dbg(f"Found commands dir was: {found_commands_dir}")
    return None


def _iter_cmd_modules(package: str, module_prefix: str = "_cmd") -> Iterable[Tuple[str, Any]]:
    """
    Yield-eli a (module_name, module) párokat a package alatti _cmd* modulokra.
    Modulnév szerint rendezünk -> determinisztikus.
    """
    pkg = importlib.import_module(package)

    module_names = []
    for modinfo in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        short = modinfo.name.rsplit(".", 1)[-1]
        if short.startswith(module_prefix):
            module_names.append(modinfo.name)

    for module_name in sorted(module_names):
        yield module_name, importlib.import_module(module_name)


def _bind(host: Any, fn: Any) -> Any:
    """Bind function as method to host instance."""
    return types.MethodType(fn, host)


def attach_cmd_functions(
    host: Any,
    *,
    package_map: Optional[Dict[str, str]] = None,
    local_subpackage: str = "commands",
    overwrite: bool = True,
    debug: bool = False,
) -> None:
    """
    Load commands from commands_package + optional extra packages.

    Conventions:
      - Command modules: _cmd_*.py
      - Command functions: top-level functions not starting with '_' and not '__*'
        -> attached to the *instance* as self._cmd_<name>(...)
      - Helper functions: top-level functions named '__*' (and always include self)
        -> attached to the *class* as private (mangled) method, so self.__helper works

    Later packages override earlier ones if overwrite=True.
    """

    def dbg(msg: str) -> None:
        if debug:
            print(f"[registry][debug] {msg}")

    cls = host.__class__

    packages = []

    local_pkg = _find_local_commands_package(host, local_subpackage=local_subpackage)
    if local_pkg:
        packages.append(local_pkg)

    dbg(f"Found local commands package: {local_pkg}")
    dbg(f"Value of local package: {local_pkg}")
    if package_map:
        packages.extend(package_map.values())

    dbg(f"Host class: {cls.__name__}")
    dbg(f"Packages to load (in order): {packages}")

    # Track what we set, for debugging/override clarity
    for pkg in packages:
        dbg(f"Loading package: {pkg}")

        for module_name, module in _iter_cmd_modules(pkg):
            dbg(f"  Import module: {module_name}")
            dbg(f"  Path: {module.__file__}")

            for name, fn in inspect.getmembers(module, inspect.isfunction):
                # Only functions defined in this module
                if fn.__module__ != module.__name__:
                    continue

                # Only attach actual command entry-points.
                # Helpers like "__check_valid_prim" must NOT be bound (they
                # don't accept self).

                if (not overwrite) and hasattr(host, name):
                    raise RuntimeError(
                        f"Command collision: {name} already exists (overwrite=False)"
                    )

                setattr(host, name, _bind(host, fn))
                dbg(f"    command -> instance.{name}  (from {module_name}.{name})")
