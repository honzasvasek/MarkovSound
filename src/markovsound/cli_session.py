"""Manage isolated MarkovSound sessions under sessions/."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .config import LEGACY_SESSION, Paths, validate_session_name


def _ensure_session_dirs(session_root: Path) -> None:
    for path in (
        session_root / "state",
        session_root / "Audio",
        session_root / "Audio" / "absorb",
        session_root / "Audio" / "queue",
        session_root / "Audio" / "staging",
        session_root / "Audio" / "archive",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _session_root(paths: Paths, name: str) -> Path:
    validate_session_name(name)
    return paths.sessions_dir / name


def _listed_session_names(paths: Paths) -> list[str]:
    names: list[str] = []
    if not paths.sessions_dir.exists():
        return names
    for entry in sorted(paths.sessions_dir.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        try:
            validate_session_name(entry.name)
        except ValueError:
            continue
        names.append(entry.name)
    return names


def _safe_move_existing(paths: Paths, kind: str) -> None:
    src = paths.repo_root / kind
    if src.is_symlink() or not src.exists():
        return
    legacy_root = _session_root(paths, LEGACY_SESSION)
    dest = legacy_root / kind
    if dest.exists():
        backup = legacy_root / f"{kind}.pre_symlink_backup"
        suffix = 1
        while backup.exists():
            suffix += 1
            backup = legacy_root / f"{kind}.pre_symlink_backup{suffix}"
        shutil.move(str(src), str(backup))
        print(f"bestaande root {kind}/ verplaatst naar {backup}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    print(f"bestaande root {kind}/ verplaatst naar {dest}")


def _replace_symlink(link: Path, target: Path) -> None:
    if link.exists() and not link.is_symlink():
        raise SystemExit(f"kan geen symlink maken; bestaat al als echte map/bestand: {link}")
    rel_target = os.path.relpath(target, start=link.parent)
    tmp = link.with_name(f".{link.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(rel_target, target_is_directory=True)
    os.replace(tmp, link)


def _write_current(paths: Paths, name: str) -> None:
    paths.current_session_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.current_session_path.with_name(f".{paths.current_session_path.name}.tmp")
    tmp.write_text(name + "\n", encoding="utf-8")
    os.replace(tmp, paths.current_session_path)


def _activate_session(paths: Paths, name: str) -> None:
    root = _session_root(paths, name)
    # Move an old pre-session checkout out of the root *before* creating the
    # target scaffolding.  Otherwise `use legacy` would create empty
    # sessions/legacy/{state,Audio} first and then treat the real old root dirs
    # as collisions, shunting them into *.pre_symlink_backup instead of making
    # them the active legacy session.
    _safe_move_existing(paths, "state")
    _safe_move_existing(paths, "Audio")
    _ensure_session_dirs(root)
    _replace_symlink(paths.repo_root / "state", root / "state")
    _replace_symlink(paths.repo_root / "Audio", root / "Audio")
    _write_current(paths, name)


def _copy_tree(src: Path, dest: Path, *, force: bool) -> None:
    if not src.exists():
        dest.mkdir(parents=True, exist_ok=True)
        return
    if dest.exists() and force:
        shutil.rmtree(dest)
    if dest.exists():
        raise SystemExit(f"bestemming bestaat al: {dest}. gebruik --force om te overschrijven.")
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("now_playing.json", "__pycache__"))


def cmd_list(_args) -> int:
    paths = Paths.discover()
    for name in _listed_session_names(paths):
        marker = "*" if name == paths.session_name else " "
        print(f"{marker} {name}\t{_session_root(paths, name)}")
    return 0


def cmd_current(_args) -> int:
    paths = Paths.discover()
    print(paths.session_name)
    print(paths.session_root)
    print(f"state -> {paths.state_dir.resolve() if paths.state_dir.exists() else paths.state_dir}")
    print(f"Audio -> {paths.audio_dir.resolve() if paths.audio_dir.exists() else paths.audio_dir}")
    return 0


def cmd_new(args) -> int:
    paths = Paths.discover()
    name = validate_session_name(args.name)
    root = _session_root(paths, name)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"sessie bestaat al: {name}")
    _ensure_session_dirs(root)
    print(f"nieuwe sessie: {name} -> {root}")
    if args.use:
        _activate_session(paths, name)
        print(f"actief: {name}")
    return 0


def cmd_use(args) -> int:
    paths = Paths.discover()
    name = validate_session_name(args.name)
    root = _session_root(paths, name)
    if not root.exists():
        print(f"nieuwe sessiemappen aangemaakt: {root}")
    _activate_session(paths, name)
    print(f"actief: {name}")
    return 0


def cmd_save(args) -> int:
    paths = Paths.discover()
    dest_name = validate_session_name(args.name)
    dest_root = _session_root(paths, dest_name)
    if dest_root.resolve() == paths.session_root.resolve():
        raise SystemExit("kan de actieve sessie niet over zichzelf opslaan")
    _copy_tree(paths.state_dir, dest_root / "state", force=args.force)
    _copy_tree(paths.audio_dir, dest_root / "Audio", force=args.force)
    print(f"sessie opgeslagen als: {dest_name} -> {dest_root}")
    if args.use:
        _activate_session(paths, dest_name)
        print(f"actief: {dest_name}")
    return 0


def cmd_clone(args) -> int:
    paths = Paths.discover()
    src_name = validate_session_name(args.source)
    dest_name = validate_session_name(args.dest)
    if src_name == dest_name:
        raise SystemExit("bron en doel moeten verschillende sessies zijn")
    src_root = _session_root(paths, src_name)
    dest_root = _session_root(paths, dest_name)
    if not src_root.exists():
        raise SystemExit(f"bron-sessie bestaat niet: {src_name}")
    if dest_root.resolve() == paths.session_root.resolve():
        raise SystemExit("kan niet over de actieve sessie heen klonen")
    _copy_tree(src_root / "state", dest_root / "state", force=args.force)
    _copy_tree(src_root / "Audio", dest_root / "Audio", force=args.force)
    print(f"sessie gekloond: {src_name} -> {dest_name}")
    if args.use:
        _activate_session(paths, dest_name)
        print(f"actief: {dest_name}")
    return 0


def cmd_delete(args) -> int:
    paths = Paths.discover()
    name = validate_session_name(args.name)
    root = _session_root(paths, name)
    if name == paths.session_name:
        raise SystemExit("kan de actieve sessie niet verwijderen. activeer eerst een andere sessie.")
    if not root.exists():
        raise SystemExit(f"sessie bestaat niet: {name}")
    if not args.force:
        typed = input(f"type '{name}' om sessie {name} definitief te verwijderen: ").strip()
        if typed != name:
            raise SystemExit("verwijderen afgebroken")
    shutil.rmtree(root)
    print(f"sessie verwijderd: {name}")
    return 0


def cmd_rename(args) -> int:
    paths = Paths.discover()
    old_name = validate_session_name(args.old)
    new_name = validate_session_name(args.new)
    if old_name == new_name:
        raise SystemExit("oude en nieuwe sessienaam moeten verschillen")
    old_root = _session_root(paths, old_name)
    new_root = _session_root(paths, new_name)
    if not old_root.exists():
        raise SystemExit(f"sessie bestaat niet: {old_name}")
    if new_root.exists():
        raise SystemExit(f"doel-sessie bestaat al: {new_name}")
    active = old_name == paths.session_name
    shutil.move(str(old_root), str(new_root))
    if active:
        _replace_symlink(paths.repo_root / "state", new_root / "state")
        _replace_symlink(paths.repo_root / "Audio", new_root / "Audio")
        _write_current(paths, new_name)
    print(f"sessie hernoemd: {old_name} -> {new_name}")
    return 0


def cmd_path(args) -> int:
    paths = Paths.discover()
    root = _session_root(paths, validate_session_name(args.name)) if args.name else paths.session_root
    print(root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="beheer MarkovSound-sessies")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="toon beschikbare sessies").set_defaults(func=cmd_list)
    sub.add_parser("current", help="toon actieve sessie").set_defaults(func=cmd_current)

    p = sub.add_parser("new", help="maak een lege sessie")
    p.add_argument("name")
    p.add_argument("--use", action="store_true", help="maak deze sessie meteen actief")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("use", help="maak een sessie actief")
    p.add_argument("name")
    p.set_defaults(func=cmd_use)

    p = sub.add_parser("save", help="kopieer de actieve sessie naar sessions/NAME")
    p.add_argument("name")
    p.add_argument("--force", action="store_true")
    p.add_argument("--use", action="store_true")
    p.set_defaults(func=cmd_save)

    p = sub.add_parser("clone", help="kopieer een bestaande sessie")
    p.add_argument("source")
    p.add_argument("dest")
    p.add_argument("--force", action="store_true")
    p.add_argument("--use", action="store_true")
    p.set_defaults(func=cmd_clone)

    p = sub.add_parser("delete", aliases=["del", "rm"], help="verwijder een niet-actieve sessie")
    p.add_argument("name")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("rename", help="hernoem een sessie")
    p.add_argument("old")
    p.add_argument("new")
    p.set_defaults(func=cmd_rename)

    p = sub.add_parser("path", help="toon pad van actieve of genoemde sessie")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
