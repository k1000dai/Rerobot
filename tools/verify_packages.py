#!/usr/bin/env python3
"""Package every crate and verify the exact archives against local sibling archives.

Cargo's built-in archive verification resolves every path dependency from crates.io.
None of these four crates has been published, so for `rerobot-train` and
`rerobot-cli` -- whose manifests depend on their siblings -- that resolution simply
fails: there is no `rerobot-core 0.1.0` on the registry to resolve to. Cargo cannot
verify a workspace whose members depend on each other until the whole set is released,
which is a chicken-and-egg problem rather than a fault in the archives.

So this script does the verification Cargo cannot: it asks Cargo to build the normal
publishable archives, extracts those exact archives, patches crates.io *only inside a
temporary verification workspace* to point at the extracted siblings, and runs their
tests and doctests there. Nothing from the source checkout is used after archive
creation, so what is tested is what would be published.

It also checks each archive's contents: every one must carry `LICENSE` and `NOTICE`,
because the READMEs tell recipients to consult them and `NOTICE` holds the LeRobot
attribution Apache-2.0 section 4(d) requires be retained.
"""

from __future__ import annotations

import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

CRATES = (
    "rerobot-core",
    "rerobot-compat",
    "rerobot-hardware",
    "rerobot-train",
    "rerobot-cli",
)

# Files every published archive must carry, whatever else it holds.
#
# `LICENSE` is Apache-2.0 and its appendix points at `NOTICE`; `NOTICE` carries the
# upstream LeRobot attribution that Apache-2.0 section 4(d) requires a redistribution
# to retain. `license = "Apache-2.0"` in the manifest is metadata, not the text, and a
# packaged README that says "see the repository root" is useless to someone holding
# only the archive.
REQUIRED_DOCUMENTS = ("LICENSE", "NOTICE")
LOCAL_DEPENDENCIES = {
    "rerobot-core": (),
    "rerobot-compat": (),
    "rerobot-hardware": (),
    "rerobot-train": ("rerobot-core",),
    "rerobot-cli": (
        "rerobot-core",
        "rerobot-compat",
        "rerobot-hardware",
        "rerobot-train",
    ),
}
ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def workspace_version() -> str:
    manifest = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', manifest, re.MULTILINE)
    if match is None:
        raise RuntimeError("workspace package version is absent")
    return match.group(1)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as package:
        for member in package.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"archive path escapes destination: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"package archive contains a link: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"package archive contains a special file: {member.name}")
        package.extractall(destination)


def check_required_documents(name: str, archive: Path) -> None:
    """Every archive must carry `LICENSE` and `NOTICE`, with content.

    Checked against the archive's own member list rather than the checkout, because the
    archive is what gets published, and an empty or truncated copy would satisfy a
    presence check while failing the obligation.
    """
    with tarfile.open(archive, "r:gz") as handle:
        members = {
            member.name.split("/", 1)[1]: member
            for member in handle.getmembers()
            if "/" in member.name
        }
    for document in REQUIRED_DOCUMENTS:
        member = members.get(document)
        if member is None:
            raise RuntimeError(
                f"{name}: {archive.name} does not carry {document}. Apache-2.0 section 4(d) "
                f"requires the NOTICE to travel with a redistribution, and the packaged README "
                f"tells recipients to consult these files."
            )
        if member.size == 0:
            raise RuntimeError(f"{name}: {archive.name} carries an empty {document}")
    print(f"  {name}: archive carries {' and '.join(REQUIRED_DOCUMENTS)}")


def main() -> None:
    version = workspace_version()
    run(
        "cargo",
        "package",
        "--workspace",
        "--allow-dirty",
        "--locked",
        "--no-verify",
    )

    with tempfile.TemporaryDirectory(prefix="rerobot-package-verify-") as temporary:
        extracted = Path(temporary)
        package_dirs: dict[str, Path] = {}
        for name in CRATES:
            archive = ROOT / "target" / "package" / f"{name}-{version}.crate"
            if not archive.is_file():
                raise RuntimeError(f"cargo did not create {archive}")
            check_required_documents(name, archive)
            safe_extract(archive, extracted)
            package_dirs[name] = extracted / f"{name}-{version}"

        for name, directory in package_dirs.items():
            dependencies = LOCAL_DEPENDENCIES[name]
            if not dependencies:
                continue
            patch_lines = ["", "[patch.crates-io]"]
            for dependency in dependencies:
                escaped = str(package_dirs[dependency]).replace("\\", "\\\\")
                patch_lines.append(f'{dependency} = {{ path = "{escaped}" }}')
            manifest = directory / "Cargo.toml"
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(patch_lines) + "\n")

        # Verify the publishable default artifact here. `rerobot-train` has an
        # explicit `cuda` feature, but these hosted package runners do not have
        # nvcc; the feature is validated separately on a CUDA host rather than
        # making archive verification pretend that CPU-only CI can exercise it.
        for name in CRATES:
            manifest = package_dirs[name] / "Cargo.toml"
            run(
                "cargo",
                "test",
                "--target-dir",
                str(extracted / "target"),
                "--manifest-path",
                str(manifest),
                "--all-targets",
                "--no-default-features",
                cwd=extracted,
            )
            run(
                "cargo",
                "test",
                "--target-dir",
                str(extracted / "target"),
                "--manifest-path",
                str(manifest),
                "--doc",
                "--no-default-features",
                cwd=extracted,
            )
    print("all packaged crates passed archive-only tests and doctests")


if __name__ == "__main__":
    main()
