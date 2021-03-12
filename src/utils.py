import logging
import subprocess
import codecs
import dataclasses
import re


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
        commit_id_out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
        commit_id_out.check_returncode()
        commit_id = codecs.decode(commit_id_out.stdout).strip()
        clean_tree_out = subprocess.run(["git", "status", "--porcelain"], capture_output=True)
        clean_tree_out.check_returncode()
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
    try:
        module_cmd_out = subprocess.run(["bash", "-c", "module list"],
                                        capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.warning("Failed to get linux module info")
        return []
    # Decode process output and scan for module info
    module_output = codecs.decode(module_cmd_out.stdout or module_cmd_out.stderr)
    rgx = re.compile(r"\d+\)\s+(?P<modname>\S+)\s?")
    modules = []
    for match in re.finditer(rgx, module_output):
        raw_name = match.group("modname")
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
