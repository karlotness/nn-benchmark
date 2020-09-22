import logging
import subprocess
import codecs


def set_up_logging(level, out_file):
    num_level = getattr(logging, level.upper(), None)
    if not isinstance(num_level, int):
        raise ValueError("Invalid log level: {}".format(level))
    handlers = []
    handlers.append(logging.StreamHandler())
    handlers.append(logging.FileHandler(filename=out_file, encoding="utf8"))
    logging.basicConfig(level=num_level, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def log_git_info():
    logger = logging.getLogger("envinfo")
    try:
        commit_id_out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
        commit_id = codecs.decode(commit_id_out.stdout).strip()
        clean_tree_out = subprocess.run(["git", "status", "--porcelain"], capture_output=True)
        worktree_state = "clean" if len(clean_tree_out.stdout) == 0 else "dirty"
        logger.info("Running on commit {} ({} worktree)".format(commit_id, worktree_state))
    except Exception:
        logger.exception("Failed to get information on git status")
