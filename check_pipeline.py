"""Read-only pre-submit check of a project folder's .py/.sh files. Changes nothing.

Reports:
  1. Broken local imports: `import shared_utils.X` / `from shared_utils.X import y` where the
     module file or the name no longer exists (e.g. the HBN_utils -> dataset_utils rename).
     Only modules found in the project or lab folder are checked; third-party packages are
     skipped because jobs run in several conda envs.
  2. Missing scripts: `python foo.py` in a .sh where foo.py does not exist.
  3. Hard-coded paths: absolute /dartfs* or /scratch* paths in code that do not exist, and every
     path on /dartfs-hpc/scratch (45-day purge) with the age of its oldest file.

Usage:
    python shared_utils/check_pipeline.py HCD [HBN ...] [--no-scratch-ages] [--include-legacy]
    python shared_utils/check_pipeline.py --submit HCD/submit_slurm_job_glm_beta.sh   # exit 2 = do not submit
"""
import argparse
import ast
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_PREFIXES = ("/dartfs-hpc/scratch", "/scratch")
PATH_PREFIXES = ("/dartfs/", "/dartfs-hpc/", "/scratch/")
PLACEHOLDER = re.compile(r"[{}*?$%<]")
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "site-packages"}
PURGE_DAYS = 45


def project_files(root, include_legacy=False):
    """Code files: git-tracked plus untracked-not-ignored inside a repo, else a pruned walk."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"],
                         capture_output=True, text=True)
    if out.returncode == 0:
        files = [root / f for f in out.stdout.splitlines()]
    else:
        files = []
        for d, dirs, names in os.walk(root):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
            files += [Path(d) / n for n in names]
    skip = SKIP_DIRS if include_legacy else SKIP_DIRS | {"legacy"}
    return sorted(f for f in files if f.suffix in (".py", ".sh") and f.is_file()
                  and not skip.intersection(x.lower() for x in f.relative_to(root).parts))


# ---------------------------------------------------------------- imports

def top_level_names(module_file):
    """Names a module defines at top level (also inside top-level if/try), or None if it uses
    `import *` or a module __getattr__, where names cannot be known statically."""
    try:
        tree = ast.parse(module_file.read_text(errors="replace"))
    except SyntaxError:
        return None
    names, stack = set(), list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if node.name == "__getattr__":
                return None
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    return None
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For)):
            for field in ("body", "orelse", "finalbody", "handlers"):
                stack += getattr(node, field, [])
        elif isinstance(node, ast.ExceptHandler):
            stack += node.body
    return names


def find_local(dotted, roots):
    """(found, path) for a dotted module among local roots. found is None when the top-level
    name is not a local regular package/module (i.e. third-party: not checked)."""
    parts = dotted.split(".")
    for root in roots:
        base = root / parts[0]
        if (base / "__init__.py").is_file() or base.with_suffix(".py").is_file():
            p = root
            for i, part in enumerate(parts):
                if (p / part / "__init__.py").is_file():
                    p = p / part
                elif (p / f"{part}.py").is_file() and i == len(parts) - 1:
                    return True, p / f"{part}.py"
                else:
                    return False, p / part
            return True, p / "__init__.py"
    return None, None


def literal_sys_path_dirs(tree, script):
    """Directories added with sys.path.insert/append of a string literal."""
    dirs = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("insert", "append")
                and ast.unparse(node.func.value) == "sys.path"):
            arg = node.args[-1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                dirs.append(Path(arg.value) if os.path.isabs(arg.value) else script.parent / arg.value)
    return dirs


def check_imports(script, tree, problems, local_modules=None):
    """Flag local imports whose module or name is missing; add found local module files to
    `local_modules` (a set) when given, so their own imports can be checked too."""
    extra = literal_sys_path_dirs(tree, script)
    for d in extra:
        if not d.is_dir():
            problems["sys.path folder missing"].append(f"{d}")
    roots = [script.parent, *extra, LAB_ROOT, script.parent.parent]
    names_cache = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [(node.module, [a.name for a in node.names])]
        else:
            continue
        for mod, imported in mods:
            found, path = find_local(mod, roots)
            if found is None:
                continue
            if not found:
                problems["missing module"].append(f"line {node.lineno}: {mod}")
                continue
            if local_modules is not None:
                local_modules.add(path)
            if not imported or "*" in imported:
                continue
            if path not in names_cache:
                names_cache[path] = top_level_names(path)
            defined = names_cache[path]
            if defined is None:
                continue
            for name in imported:
                is_submodule = path.name == "__init__.py" and (
                    (path.parent / f"{name}.py").is_file() or (path.parent / name / "__init__.py").is_file())
                if name not in defined and not is_submodule:
                    problems["missing name"].append(f"line {node.lineno}: from {mod} import {name}")


# ---------------------------------------------------------------- paths

def string_constants(tree):
    """String literals, plus the literal head of f-strings (up to the first {...})."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr) and node.values:
            head = node.values[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                yield node.lineno, head.value + "{"


def sh_strings(text):
    """Path-like tokens from non-comment shell lines."""
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"(?<![\w}\])])(/(?:dartfs|dartfs-hpc|scratch)/[^\s\"'`;|)]*)", line):
            yield i, m.group(1)


def existing_part(path):
    """The part of a path before any placeholder, and the folder that must exist for it."""
    m = PLACEHOLDER.search(path)
    if m is None:
        return path.rstrip("/"), path.rstrip("/")
    head = path[:m.start()]
    return head, head.rstrip("/") if head.endswith("/") else os.path.dirname(head)


def matches(head):
    """Existing paths a (possibly partial) path can refer to: itself, or, when it is the start
    of a name completed at run time (".../HCD_" + task), the entries starting with it."""
    if os.path.exists(head):
        return [head.rstrip("/")]
    parent, stem = os.path.split(head)
    if stem and os.path.isdir(parent):
        return [os.path.join(parent, e) for e in os.listdir(parent) if e.startswith(stem)]
    return []


def scratch_folder(path):
    """First folder under /dartfs-hpc/scratch/<user>/ (or /scratch/<user>/) that a path lies in."""
    for prefix in SCRATCH_PREFIXES:
        if path.startswith(prefix + "/"):
            parts = path[len(prefix) + 1:].split("/")
            return "/".join([prefix, *parts[:2]]) if len(parts) >= 2 else None
    return None


def check_paths(lines_and_strings, problems, scratch_refs, script):
    for lineno, s in lines_and_strings:
        s = s.strip()
        if not s.startswith(PATH_PREFIXES) or "\n" in s:
            continue
        head, must_exist = existing_part(s)
        found = matches(head) or ([must_exist] if os.path.exists(must_exist) else [])
        if not found:
            problems["path missing (fine if the script creates it)"].append(f"line {lineno}: {s}")
        for f in found:
            folder = scratch_folder(f)
            if folder:
                scratch_refs[folder].add(script)


def check_sh_scripts(script, text, problems, root):
    bases = [script.parent, root, LAB_ROOT]  # submit scripts often cd to the project or lab root
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"\bpython3?\s+(?:-\S+\s+)*([\w./-]+\.py)\b", line):
            target = m.group(1)
            candidates = [Path(target)] if os.path.isabs(target) else [b / target for b in bases]
            if not any(c.is_file() for c in candidates):
                problems["sh calls missing script"].append(f"line {i}: {target}")


def oldest_age_days(folder, now):
    oldest, n = now, 0
    for d, dirs, names in os.walk(folder):
        for name in names:
            try:
                oldest = min(oldest, os.lstat(os.path.join(d, name)).st_mtime)
                n += 1
            except OSError:
                pass
    return (now - oldest) / 86400, n


# ---------------------------------------------------------------- submit mode

# Problems that make a job fail at once; --submit exits 2 if any are found.
BLOCKING = ("syntax error", "missing module", "missing name", "sh calls missing script")


def sh_python_targets(sh, text, problems):
    """Existing .py files a submit script mentions (python x.py, filename='x.py', $DIR/x.py,
    freeze_scripts.sh x.py ...). Simple VAR=value assignments in the script are filled in.
    A missing literal path is recorded as blocking; one behind an unresolved variable as a
    warning, since the variable may be set at run time."""
    bases = [sh.parent, Path.cwd(), LAB_ROOT]
    known = {}
    lines = [l for l in text.splitlines()]
    for line in lines:
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)=(['\"]?)([^'\"\s;`]*)\2\s*(?:#.*)?$", line)
        if m:
            value = re.sub(r"\$\{?(\w+)\}?", lambda v: known.get(v.group(1), v.group(0)), m.group(3))
            if "$" not in value:
                known[m.group(1)] = value
    targets = set()
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"(?<![\w.-])([\w./${}-]*\.py)\b", line):
            raw = re.sub(r"\$\{?(\w+)\}?", lambda v: known.get(v.group(1), v.group(0)), m.group(1))
            unresolved = "$" in raw
            name = re.split(r"\$\{?\w+\}?/?", raw)[-1] if unresolved else raw
            if not name.endswith(".py") or name == ".py":
                continue
            candidates = [Path(name)] if os.path.isabs(name) else [b / name for b in bases]
            hit = next((c for c in candidates if c.is_file()), None)
            if hit is not None:
                targets.add(hit.resolve())
            elif unresolved:
                problems["script behind unresolved variable"].append(f"line {i}: {m.group(1)}")
            else:
                problems["sh calls missing script"].append(f"line {i}: {name}")
    return targets


def check_py(f, problems, local_modules):
    try:
        tree = ast.parse(f.read_text(errors="replace"))
    except SyntaxError as e:
        kind = ("needs Python >= 3.12" if sys.version_info < (3, 12) and "f-string" in e.msg
                else "syntax error")
        problems[kind].append(f"line {e.lineno}: {e.msg}")
        return
    check_imports(f, tree, problems, local_modules)


def check_submit(paths):
    """Check only what a submission runs: the given .sh/.py files, the scripts each .sh calls,
    and the local modules (e.g. shared_utils) they import, recursively. Returns 2 if anything
    in BLOCKING was found, else 0."""
    by_file, py_todo = {}, set()
    for p in paths:
        p = Path(p).resolve()
        problems = defaultdict(list)
        if not p.is_file():
            problems["sh calls missing script"].append("file not found")
        elif p.suffix == ".py":
            py_todo.add(p)
        else:
            py_todo |= sh_python_targets(p, p.read_text(errors="replace"), problems)
        if problems:
            by_file[p] = problems
    done = set()
    while py_todo:
        f = py_todo.pop()
        done.add(f)
        problems, found = defaultdict(list), set()
        check_py(f, problems, found)
        py_todo |= {m.resolve() for m in found if m.resolve().is_relative_to(LAB_ROOT)} - done
        if problems:
            by_file.setdefault(f, defaultdict(list)).update(problems)
    blocking = False
    for f, probs in by_file.items():
        for kind, msgs in probs.items():
            blocking |= kind in BLOCKING
            tag = "BLOCK" if kind in BLOCKING else "warn"
            for msg in msgs:
                print(f"{tag}  {f}  [{kind}] {msg}")
    print(f"checked {len(done)} .py file(s): {'BLOCKED' if blocking else 'ok'}")
    return 2 if blocking else 0


# ---------------------------------------------------------------- main

def check_project(root, scratch_ages, include_legacy=False):
    root = root.resolve()
    by_file, scratch_refs = {}, defaultdict(set)
    files = project_files(root, include_legacy)
    for f in files:
        problems = defaultdict(list)
        text = f.read_text(errors="replace")
        if f.suffix == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError as e:
                # Same-quote nesting inside f-strings is valid only from Python 3.12 (PEP 701).
                kind = ("needs Python >= 3.12" if sys.version_info < (3, 12) and "f-string" in e.msg
                        else "syntax error")
                problems[kind].append(f"line {e.lineno}: {e.msg}")
            else:
                check_imports(f, tree, problems)
                check_paths(string_constants(tree), problems, scratch_refs, f)
        else:
            check_sh_scripts(f, text, problems, root)
            check_paths(sh_strings(text), problems, scratch_refs, f)
        if problems:
            by_file[f.relative_to(root)] = problems

    print(f"=== {root}  ({len(files)} .py/.sh files)")
    order = ["syntax error", "needs Python >= 3.12", "missing module", "missing name", "sh calls missing script",
             "sys.path folder missing", "path missing (fine if the script creates it)"]
    for kind in order:
        hits = [(f, msg) for f, probs in by_file.items() for msg in probs.get(kind, [])]
        if hits:
            print(f"\n[{kind}]  {len(hits)}")
            for f, msg in hits:
                print(f"  {f}  {msg}")

    if scratch_refs:
        tops = sorted(scratch_refs)
        print(f"\n[on purgeable scratch]  {len(tops)} folder(s); purge is {PURGE_DAYS} days "
              "(mtime shown; RC has not confirmed which clock they purge on)")
        now = time.time()
        for p in tops:
            users = sorted(str(s.relative_to(root)) for s in scratch_refs[p])
            age = ""
            if scratch_ages:
                days, n = oldest_age_days(p, now) if os.path.isdir(p) else (
                    (now - os.lstat(p).st_mtime) / 86400, 1)
                flag = "  <-- past purge age" if days > PURGE_DAYS else ""
                age = (f"  oldest file {days:.0f} d ({n} files){flag}" if n
                       else "  empty folder")
            print(f"  {p}{age}")
            print(f"      used by: {', '.join(users[:6])}{' ...' if len(users) > 6 else ''}")
    if not by_file and not scratch_refs:
        print("  no problems found")
    return by_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", type=Path,
                    help="project folders, or with --submit the .sh/.py files being submitted")
    ap.add_argument("--submit", action="store_true",
                    help="check only the given submit scripts, what they run and what that "
                         "imports; exit 2 on a problem that makes the job fail at once")
    ap.add_argument("--no-scratch-ages", action="store_true",
                    help="skip walking scratch folders for file ages (faster)")
    ap.add_argument("--include-legacy", action="store_true",
                    help="also check folders named legacy/ (skipped by default: retired code)")
    args = ap.parse_args()
    if args.submit:
        sys.exit(check_submit(args.folders))
    for folder in args.folders:
        check_project(folder, scratch_ages=not args.no_scratch_ages,
                      include_legacy=args.include_legacy)
        print()


if __name__ == "__main__":
    main()
