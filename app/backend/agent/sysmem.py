"""Free commit memory (RAM + page file). This PC runs close to its commit limit, so the app
checks it before starting anything big (the Claude Code CLI, a run_python child)."""
import ctypes
import os


def free_commit_mb():
    """Commit memory still free in MB, or None off Windows."""
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_commit", "avail_commit",
                                              "total_virtual", "avail_virtual", "avail_ext_virtual")]

    st = MemoryStatusEx(dwLength=ctypes.sizeof(MemoryStatusEx))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.avail_commit // 2**20
