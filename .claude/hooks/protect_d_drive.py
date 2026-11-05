#!/usr/bin/env python3
"""PreToolUse hook: block Bash commands that could change files on the D: drive.

D: holds the only copies of my raw dive footage. Reading it is fine (frame extraction does
that); anything that writes, moves, deletes, or copies raw footage is blocked unless every D:
path in the command is inside D:\\underwater_tagger_extracted\\.

This is a text check on the command, so it's a safety net, not a sandbox: a Python script that
writes to D: wouldn't be caught. The edit-tool deny rule in .claude/settings.json covers
Claude's Edit/Write tools.

Exit code 2 blocks the command and shows stderr to Claude.
"""

import json
import posixpath
import re
import sys

ALLOWED = "/mnt/d/underwater_tagger_extracted"

# Commands and flags that write, move, delete, or copy files (bash and Windows shells).
WRITE_WORDS = re.compile(
    r"(?<![\w-])("
    r"rm|rmdir|mv|cp|rsync|touch|mkdir|truncate|shred|chmod|chown|ln|dd|tee|install|"
    r"unzip|7z|tar|zip|sed\s+-i|perl\s+-i|"
    r"del|erase|move|copy|ren|rename|robocopy|xcopy|"
    r"Remove-Item|Move-Item|Copy-Item|Rename-Item|Set-Content|Out-File|New-Item"
    r")(?![\w-])",
    re.IGNORECASE,
)
REDIRECT = re.compile(r">>?\s*['\"]?(/mnt/d/[^\s'\";|&]*)", re.IGNORECASE)
POSIX_PATH = re.compile(r"/mnt/d(?:/[^\s'\";|&]*)?", re.IGNORECASE)
WINDOWS_PATH = re.compile(r"\b[dD]:[\\/][^\s'\";|&]*")


def to_posix(path: str) -> str:
    if re.match(r"[dD]:", path):
        path = "/mnt/d" + path[2:].replace("\\", "/")
    return posixpath.normpath(path).lower()


def is_allowed(path: str) -> bool:
    p = to_posix(path)
    return p == ALLOWED or p.startswith(ALLOWED + "/")


def check(command: str) -> str | None:
    """Return a reason to block, or None to allow."""
    for target in REDIRECT.findall(command):
        if not is_allowed(target):
            return f"redirects output into {target}"
    d_paths = POSIX_PATH.findall(command) + WINDOWS_PATH.findall(command)
    outside = [p for p in d_paths if not is_allowed(p)]
    if outside and WRITE_WORDS.search(command):
        verb = WRITE_WORDS.search(command).group(0)
        return f"uses '{verb}' with D: path(s) outside the extracted folder: {', '.join(outside)}"
    return None


def main() -> None:
    event = json.load(sys.stdin)
    command = event.get("tool_input", {}).get("command", "")
    reason = check(command)
    if reason:
        print(
            f"Blocked: this command {reason}. D: holds the only copies of the raw footage; "
            f"only {ALLOWED} is writable, and footage must not be copied off the drive. "
            "Read-only commands are fine. If this is a false positive, ask the user to run it.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
