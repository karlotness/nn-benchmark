import logging
import subprocess
import codecs
import dataclasses
import os


def set_up_logging(level, out_file):
    num_level = getattr(logging, level.upper(), None)
    if not isinstance(num_level, int):
        raise ValueError("Invalid log level: {}".format(level))
    handlers = []
    handlers.append(logging.StreamHandler())
    handlers.append(logging.FileHandler(filename=out_file, encoding="utf8"))
    logging.basicConfig(level=num_level, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@dataclasses.dataclass
class CommitInfo:
    hash: str
    clean_worktree: bool


def get_git_info(base_logger=None):
    if base_logger:
        logger = base_logger.getChild("gitinfo")
    else:
        logger = logging.getLogger("gitinfo")
    try:
        commit_id_out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True)
        commit_id = codecs.decode(commit_id_out.stdout).strip()
        clean_tree_out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, check=True)
        clean_worktree = len(clean_tree_out.stdout) == 0
        return CommitInfo(hash=commit_id, clean_worktree=clean_worktree)
        logger.info("Running on commit {} ({} worktree)".format(commit_id,
                                                                "clean" if clean_worktree else "dirty"))
    except Exception:
        logger.exception("Failed to get information on git status")
        return None


@dataclasses.dataclass
class ModuleInfo:
    name: str
    version: str
    raw_name: str


def get_loaded_modules(base_logger=None):
    if base_logger:
        logger = base_logger.getChild("moduleinfo")
    else:
        logger = logging.getLogger("moduleinfo")

    module_list = os.environ.get("LOADEDMODULES", None)
    if module_list is None:
        logger.warning("Failed to get linux module info")
        return []
    modules = []
    for raw_name in module_list.strip().split(":"):
        # Split components of module name
        if "/" in raw_name:
            components = raw_name.split("/")
        else:
            components = raw_name.split("-")
        # Extract components
        name = components[0]
        if len(components) > 1:
            version = components[-1]
        else:
            version = ""
        modules.append(ModuleInfo(name=name, version=version,
                                  raw_name=raw_name))
    return modules
