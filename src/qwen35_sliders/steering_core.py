"""Read the shared steering source without replacing the installed RAW owner."""
from hashlib import sha256
from sys import modules
from threading import Lock
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
_SOURCE = {
    "adaptive_steering": (10329, "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"),
    "prompt_steering": (3919, "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"),
}
_LOCK = Lock()
_core = None


def load_steering_core(wheel):
    """Verify the exact #269 wheel members before loading their definitions."""
    global _core
    sources = {}
    with ZipFile(wheel) as archive:
        for name, (size, digest) in _SOURCE.items():
            path = f"particle_sliders/{name}.py"
            entry = archive.getinfo(path)
            if entry.file_size != size:
                raise ValueError(f"Shared steering source size differs: {path}")
            source = archive.read(entry)
            if sha256(source).hexdigest() != digest:
                raise ValueError(f"Shared steering source checksum differs: {path}")
            sources[name] = source
    with _LOCK:
        if _core is None:
            loaded = {}
            try:
                for name, source in sources.items():
                    # Dataclasses resolve annotations through their registered module.
                    namespace = f"{__name__}._{name}_{id(_LOCK)}"
                    module = ModuleType(namespace)
                    module.__file__ = f"{wheel}/particle_sliders/{name}.py"
                    loaded[name] = modules[namespace] = module
                    exec(compile(source, module.__file__, "exec"), module.__dict__)
            except BaseException:
                for module in loaded.values():
                    del modules[module.__name__]
                raise
            _core = SimpleNamespace(adaptive=loaded["adaptive_steering"], prompt=loaded["prompt_steering"])
        return _core
