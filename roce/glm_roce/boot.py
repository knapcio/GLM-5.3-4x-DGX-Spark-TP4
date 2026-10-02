"""Env-gated start-up hook, imported from ``glm_roce.pth`` in every Python process.

With ``GLM_ROCE_ALLREDUCE`` unset or not "1" this module does nothing else: no
import hook, no b12x on ``sys.path``, the image behaves like its base.

With ``GLM_ROCE_ALLREDUCE=1`` it
- puts the vendored b12x (``GLM_ROCE_B12X_PATH``, default ``/opt/glm-roce-b12x``)
  first on ``sys.path``, and
- installs a post-import hook that applies ``glm_roce.install.PATCHES`` to each vLLM
  module right after that module executes, in whichever process imports it (API
  server, engine core, the spawned TP workers).

A patch that fails raises: the operator asked for RoCEnante, so a half-patched
engine must not come up.  ``verify_targets`` in the image build catches drift
earlier.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import threading

DEFAULT_B12X_PATH = "/opt/glm-roce-b12x"


class PostImportPatcher(importlib.abc.MetaPathFinder):
    """Run ``patch(module)`` right after each target module executes (once per name)."""

    def __init__(self, targets):
        self._targets = dict(targets)
        self._busy = threading.local()

    def pending(self):
        return sorted(self._targets)

    def find_spec(self, name, path=None, target=None):
        if name not in self._targets:
            return None
        busy = getattr(self._busy, "names", None)
        if busy is None:
            busy = self._busy.names = set()
        if name in busy:
            return None  # our own find_spec call below
        busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            busy.discard(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        patch = self._targets.pop(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                patch(module)
            except Exception as exc:
                raise RuntimeError(f"GLM_ROCE: patching {name} failed: {exc!r}") from exc

        spec.loader.exec_module = exec_and_patch
        return spec

    def patch_already_imported(self):
        """Apply patches for targets imported before the hook existed."""
        for name in list(self._targets):
            module = sys.modules.get(name)
            if module is not None:
                self._targets.pop(name)(module)


def install_hooks(targets, *, b12x_path=None):
    """Install the post-import patcher (and the b12x path); returns the finder."""
    if b12x_path and os.path.isdir(b12x_path) and b12x_path not in sys.path:
        sys.path.insert(0, b12x_path)
    finder = PostImportPatcher(targets)
    sys.meta_path.insert(0, finder)
    finder.patch_already_imported()
    return finder


def main():
    from glm_roce import enabled

    if not enabled():
        return None
    from glm_roce.install import PATCHES

    return install_hooks(
        PATCHES, b12x_path=os.environ.get("GLM_ROCE_B12X_PATH", DEFAULT_B12X_PATH)
    )


_finder = main()
