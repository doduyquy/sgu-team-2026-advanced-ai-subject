"""Windows per-user DPAPI storage and scoped Kaggle CLI authentication."""
import ctypes
from ctypes import wintypes
from getpass import getpass, GetPassWarning
import os
from pathlib import Path
import re
import shutil
import subprocess
import warnings

from cloud.kaggle.common import ROOT, read, write

SLOTS = tuple(f"account_{i:02}" for i in range(1, 9))
OWNER = re.compile(r"[a-z0-9_-]{1,40}")

class Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

def home():
    if os.name != "nt" or "LOCALAPPDATA" not in os.environ:
        raise RuntimeError("Credential setup requires a local Windows user session")
    directory = (Path(os.environ["LOCALAPPDATA"]) / "XeTuHanh/Kaggle").resolve()
    if directory.is_relative_to(ROOT.parent):
        raise RuntimeError("Credential storage must be outside the checkout")
    return directory

def dpapi(data, encrypt):
    source = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    incoming = Blob(len(data), source)
    outgoing = Blob()
    api = ctypes.windll.crypt32
    operation = api.CryptProtectData if encrypt else api.CryptUnprotectData
    operation.restype = wintypes.BOOL
    if not operation(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise RuntimeError("Windows DPAPI operation failed")
    free = ctypes.windll.kernel32.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        ctypes.memset(outgoing.data, 0, outgoing.size)
        free(outgoing.data)

def slot(alias):
    directory = home() / alias
    return directory, read(directory / "slot.json") if (directory / "slot.json").exists() else None

def cli(alias, args, timeout=90):
    directory, record = slot(alias)
    if not record or not record.get("authorized"):
        raise RuntimeError("Slot missing owner authorization")
    exe = shutil.which("kaggle")
    if not exe:
        raise RuntimeError("Kaggle CLI is unavailable")
    config = directory / "cli"
    config.mkdir(exist_ok=True)
    if any(config.iterdir()):
        raise RuntimeError("Scoped Kaggle config directory must be empty")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("KAGGLE_") and k.upper() != "KGAT_TOKEN"}
    env["KAGGLE_CONFIG_DIR"] = str(config)
    token = dpapi((directory / "token.dpapi").read_bytes(), False).decode("utf-8")
    env["KAGGLE_API_TOKEN"] = token
    try:
        result = subprocess.run([exe, *args], env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, ""
    # Callers may extract only validated owner, version and status fields.
    return result.returncode, result.stdout.replace(token, "<redacted>")

def verify_owner(alias):
    _, record = slot(alias)
    if not record:
        return False
    code, output = cli(alias, ["config", "view"])
    users = re.findall(r"^\s*-?\s*username\s*:\s*([a-z0-9_-]+)\s*$", output, re.I | re.M)
    token_auth = re.search(r"^\s*-?\s*auth_method\s*:\s*ACCESS_TOKEN\s*$", output, re.M)
    return code == 0 and users == [record["owner"]] and bool(token_auth)

def setup(alias, owner, minutes, concurrency):
    if not OWNER.fullmatch(owner) or not 0 <= minutes <= 10080 or not 0 <= concurrency <= 2:
        raise ValueError("Invalid owner or observed quota/concurrency")
    if input("Chủ tài khoản xác nhận quyền sử dụng slot này? Gõ YES: ") != "YES":
        raise ValueError("Owner authorization is required")
    directory, record = slot(alias)
    if record and record["owner"] != owner:
        raise ValueError("Use a different slot for a different account")
    with warnings.catch_warnings():
        warnings.simplefilter("error", GetPassWarning)
        token = getpass("Kaggle API token (ẩn): ").strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("A valid API token is required")
    from datetime import datetime, timezone
    directory.mkdir(parents=True, exist_ok=True)
    encrypted = directory / "token.tmp"
    encrypted.write_bytes(dpapi(token.encode(), True))
    encrypted.replace(directory / "token.dpapi")
    write(directory / "slot.json", {"owner": owner, "authorized": True, "credential": "Windows user DPAPI",
                                   "auth": "api_token", "gpu_minutes_observed": minutes,
                                   "concurrency_observed": concurrency,
                                   "observed_at": datetime.now(timezone.utc).isoformat()})
    print("IDENTITY_VERIFIED" if verify_owner(alias) else "IDENTITY_UNVERIFIED")

