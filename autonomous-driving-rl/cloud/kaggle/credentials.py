"""Windows per-user DPAPI storage and scoped Kaggle CLI authentication."""
import ctypes
from ctypes import wintypes
import csv
from datetime import datetime, timezone
import math
from getpass import getpass, GetPassWarning
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import warnings

from cloud.kaggle.common import ROOT, read, write

SLOTS = tuple(f"account_{i:02}" for i in range(1, 9))
OWNER = re.compile(r"[a-z0-9_-]{1,40}")
TOKEN = re.compile(r"(?<![A-Za-z0-9_-])KGAT_[A-Za-z0-9_-]{15,}(?![A-Za-z0-9_-])")

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

def cli(alias, args, timeout=90, *, include_stderr=False):
    return token_cli(slot_token(alias), args, timeout, include_stderr=include_stderr)

def slot_token(alias):
    directory, record = slot(alias)
    if not record or not record.get("authorized"):
        raise RuntimeError("Slot missing owner authorization")
    return dpapi((directory / "token.dpapi").read_bytes(), False).decode("utf-8")

def token_cli(token, args, timeout=90, *, include_stderr=False):
    exe = shutil.which("kaggle")
    if not exe:
        raise RuntimeError("Kaggle CLI is unavailable")
    return token_command(token, [exe, "-W", *args], timeout, include_stderr=include_stderr)

def token_command(token, command, timeout, *, include_stderr=False):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("KAGGLE_") and k.upper() != "KGAT_TOKEN"}
    env["KAGGLE_API_TOKEN"] = token
    with TemporaryDirectory(prefix="stage5-kaggle-cli-") as config:
        env["KAGGLE_CONFIG_DIR"] = config
        try:
            result = subprocess.run(command, env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return -1, ""
    # Callers may extract only validated owner, version and status fields.
    output = result.stdout + ("\n" + result.stderr if include_stderr else "")
    return result.returncode, TOKEN.sub("<redacted>", output.replace(token, "<redacted>"))

def private_input(alias, ref):
    # The dataset status CLI omits ACL. Kaggle's SDK GetDataset response includes it.
    command = [sys.executable, "-B", "-m", "cloud.kaggle.dataset_acl", ref]
    code, result = token_command(slot_token(alias), command, 25)
    return code == 0 and result.splitlines()[-1:] == ["PRIVATE_VERSION_1"]

def current_private_kernel_version(alias, ref):
    command = [sys.executable, "-B", "-m", "cloud.kaggle.kernel_version", ref]
    code, result = token_command(slot_token(alias), command, 25)
    lines = result.splitlines()
    match = re.fullmatch(r"PRIVATE_KERNEL_VERSION ([1-9]\d*)", lines[-1]) if code == 0 and lines else None
    return int(match[1]) if match else None

def download_kernel_output(alias, ref, version, path):
    command = [sys.executable, "-B", "-m", "cloud.kaggle.kernel_version", "download", ref, str(version), str(path)]
    code, result = token_command(slot_token(alias), command, 180)
    return code == 0 and result.splitlines()[-1:] == ["VERSIONED_OUTPUT_READY"]

def authenticated_owner(code, output):
    users = re.findall(r"^\s*-?\s*username\s*:\s*([a-z0-9_-]+)\s*$", output, re.I | re.M)
    token_auth = re.search(r"^\s*-?\s*auth_method\s*:\s*ACCESS_TOKEN\s*$", output, re.M)
    if code == 0 and len(users) == 1 and token_auth:
        return users[0].lower()
    return None

def verify_owner(alias):
    _, record = slot(alias)
    return bool(record and authenticated_owner(*cli(alias, ["config", "view"])) == record["owner"])

def accelerator_quota(alias):
    code, output = cli(alias, ["quota", "--format", "csv"], timeout=25)
    lines = output.splitlines()
    header = next((i for i, line in enumerate(lines) if line.lower().startswith("resource,")), None)
    if code or header is None:
        return {}
    minutes = {}
    for row in csv.DictReader(lines[header:]):
        resource = row.get("resource")
        remaining = row.get("remaining") or ""
        if resource in {"GPU", "TPU"} and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?h", remaining):
            # The CLI rounds to two decimal hours. Reserve one minute for that uncertainty.
            minutes[resource] = max(0, math.floor(float(remaining[:-1]) * 60) - 1)
    return minutes

def register(alias, owner, token, minutes=None, concurrency=None):
    directory, record = slot(alias)
    if record and record["owner"] != owner:
        raise ValueError("Use a different slot for a different account")
    directory.mkdir(parents=True, exist_ok=True)
    encrypted = directory / "token.tmp"
    encrypted.write_bytes(dpapi(token.encode(), True))
    encrypted.replace(directory / "token.dpapi")
    write(directory / "slot.json", {"owner": owner, "authorized": True, "credential": "Windows user DPAPI",
                                   "auth": "api_token", "gpu_minutes_observed": minutes,
                                   "concurrency_observed": concurrency,
                                   "observed_at": datetime.now(timezone.utc).isoformat()})

def import_file(source):
    path = source.resolve(strict=True)
    if path.is_relative_to(ROOT.parent) or path.stat().st_size > 65536:
        raise ValueError("Source must be a small explicit local file outside the checkout")
    tokens = list(dict.fromkeys(TOKEN.findall(path.read_text(encoding="utf-8-sig"))))
    if len(tokens) != len(SLOTS):
        raise ValueError("Expected eight distinct Kaggle access tokens")
    existing = {record["owner"] for alias in SLOTS if (record := slot(alias)[1])}
    for alias, token in zip(SLOTS, tokens, strict=True):
        if slot(alias)[1]:
            print(alias, "EXISTING")
            continue
        owner = authenticated_owner(*token_cli(token, ["config", "view"], timeout=25))
        if not owner or owner in existing:
            print(alias, "IDENTITY_UNVERIFIED" if not owner else "DUPLICATE_OWNER")
            continue
        register(alias, owner, token)
        existing.add(owner)
        print(alias, "IDENTITY_VERIFIED")

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
    if authenticated_owner(*token_cli(token, ["config", "view"])) != owner:
        raise ValueError("Kaggle token identity does not match the slot owner")
    register(alias, owner, token, minutes, concurrency)
    print("IDENTITY_VERIFIED")

