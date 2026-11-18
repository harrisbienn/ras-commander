"""Verified HEC Monolith runtime cache; trust is never learned from its contents."""

import hashlib
import json
import logging
import os
import platform
import re
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import ClassVar

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HecMonolithDownloader:
    """Use an independently reviewed manifest, with sealed-cache defaults.

    RAS_DSS_TRUST_MANIFEST must identify JSON outside the writable cache.
    RAS_DSS_ALLOW_DOWNLOADS=1 explicitly enables verified network acquisition.
    Administrators must protect manifest/cache permissions independently.
    """

    TRUST_POLICY = "ras-commander/dss-runtime-trust/1.0.0"
    NEXUS_BASE = "https://www.hec.usace.army.mil/nexus/repository/maven-public"
    MAVEN_BASE = "https://repo.maven.apache.org/maven2"
    COMMON_JARS: ClassVar[list[dict[str, str]]] = [
        {
            "artifactId": "hec-monolith",
            "group": "mil.army.usace.hec",
            "version": "3.3.27",
        },
        {
            "artifactId": "hec-monolith-compat",
            "group": "mil.army.usace.hec",
            "version": "3.3.27",
        },
        {
            "artifactId": "hec-nucleus-data",
            "group": "mil.army.usace.hec",
            "version": "2.0.1",
        },
        {
            "artifactId": "hec-nucleus-metadata",
            "group": "mil.army.usace.hec",
            "version": "2.0.1",
        },
        {
            "artifactId": "hecnf",
            "group": "mil.army.usace.hec.hecnf",
            "version": "7.2.0",
        },
        {
            "artifactId": "flogger",
            "group": "com.google.flogger",
            "version": "0.5.1",
            "source": "maven",
        },
        {
            "artifactId": "flogger-system-backend",
            "group": "com.google.flogger",
            "version": "0.5.1",
            "source": "maven",
        },
    ]
    NATIVE_LIBS: ClassVar[dict[str, dict[str, str]]] = {
        "Windows": {
            "artifactId": "javaHeclib",
            "group": "mil.army.usace.hec",
            "version": "7-IU-8-win-x86_64",
            "extension": "zip",
        },
        "Linux": {
            "artifactId": "javaHeclib",
            "group": "mil.army.usace.hec",
            "version": "7-IU-8-linux-x86_64",
            "extension": "zip",
        },
        "Darwin": {
            "artifactId": "javaHeclib",
            "group": "mil.army.usace.hec",
            "version": "7-IU-8-macOS-x86_64",
            "extension": "zip",
        },
    }
    MAX_BYTES = 1024 * 1024 * 1024
    DOWNLOAD_SECONDS = 300
    MAX_ARCHIVE_MEMBERS = 256
    MAX_MANIFEST_BYTES = 1024 * 1024

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        show_progress: bool | None = None,
        *,
        trust_manifest: str | Path | None = None,
        allow_downloads: bool | None = None,
    ):
        """Select the approved manifest and cache without creating or acquiring files."""
        self.cache_dir = Path(
            os.path.abspath(
                cache_dir
                or os.environ.get("RAS_DSS_CACHE_DIR")
                or Path.home() / ".ras-commander" / "dss"
            )
        )
        self.jar_dir = self.cache_dir / "jar"
        self.lib_dir = self.cache_dir / "lib"
        manifest = trust_manifest or os.environ.get("RAS_DSS_TRUST_MANIFEST")
        self.trust_manifest = Path(os.path.abspath(manifest)) if manifest else None
        if allow_downloads is not None and type(allow_downloads) is not bool:
            raise TypeError("allow_downloads must be bool or None")
        if show_progress is not None and type(show_progress) is not bool:
            raise TypeError("show_progress must be bool or None")
        self.show_progress = show_progress
        self.allow_downloads = (
            os.environ.get("RAS_DSS_ALLOW_DOWNLOADS") == "1"
            if allow_downloads is None
            else allow_downloads
        )

    def _should_show_progress(self) -> bool:
        """Keep automatic progress limited to interactive stderr sessions."""
        if self.show_progress is not None:
            return self.show_progress
        isatty = getattr(sys.stderr, "isatty", None)
        return bool(isatty and isatty())

    @staticmethod
    def _is_link(path: Path) -> bool:
        """Include Windows junctions on supported Python versions before 3.12."""
        if path.is_symlink():
            return True
        if not path.exists():
            return False
        return bool(
            getattr(path.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )

    @staticmethod
    def _unique_json(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("Duplicate DSS trust manifest field")
            result[key] = value
        return result

    @staticmethod
    def _filename(artifact):
        return f"{artifact['artifactId']}-{artifact['version']}.{artifact.get('extension', 'jar')}"

    @staticmethod
    def _safe_name(name):
        if not isinstance(name, str):
            raise TypeError("Executable artifact paths must be strings")
        path = PurePosixPath(name)
        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
        }
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ":" in name
            or any(ord(character) < 32 or character in '<>"|?*' for character in name)
            or any(
                part in ("", ".", "..") or part.endswith((".", " "))
                for part in name.split("/")
            )
            or any(part.split(".")[0].upper() in reserved for part in path.parts)
        ):
            raise ValueError(f"Unsafe executable artifact path: {name!r}")
        return path

    def _trust(self):
        if self.trust_manifest is None:
            raise RuntimeError(
                "Set RAS_DSS_TRUST_MANIFEST to an independently reviewed manifest outside the executable cache"
            )
        if self.trust_manifest.resolve().is_relative_to(self.cache_dir.resolve()):
            raise ValueError(
                "The DSS trust manifest must be outside the writable executable cache"
            )
        if any(
            self._is_link(path)
            for path in (self.trust_manifest, *self.trust_manifest.parents)
        ):
            raise ValueError(
                "DSS trust manifest path must not contain links or junctions"
            )
        if not self.trust_manifest.is_file():
            raise ValueError("DSS trust manifest must be a regular file")
        with self.trust_manifest.open("rb") as stream:
            encoded = stream.read(self.MAX_MANIFEST_BYTES + 1)
        if len(encoded) > self.MAX_MANIFEST_BYTES:
            raise ValueError("DSS trust manifest exceeds its byte limit")
        manifest = json.loads(encoded, object_pairs_hook=self._unique_json)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"format_version", "platform", "artifacts", "downloads"}
            or type(manifest.get("format_version")) is not int
            or manifest.get("format_version") != 1
            or manifest.get("platform") != platform.system()
        ):
            raise ValueError(
                "DSS trust manifest version/platform does not match this runtime"
            )
        for section in ("artifacts", "downloads"):
            entries = manifest.get(section)
            if not isinstance(entries, dict) or not entries:
                raise ValueError(f"DSS trust manifest requires {section}")
            if len(entries) > self.MAX_ARCHIVE_MEMBERS:
                raise ValueError("DSS trust manifest artifact count exceeds its limit")
            seen = set()
            for name, metadata in entries.items():
                path = self._safe_name(name)
                if (
                    len(path.parts) < 2
                    or path.parts[0] not in {"jar", "lib"}
                    or name.casefold() in seen
                ):
                    raise ValueError(
                        f"Invalid or case-colliding artifact identity: {name}"
                    )
                seen.add(name.casefold())
                if (
                    not isinstance(metadata, dict)
                    or set(metadata) != {"sha256", "size_bytes"}
                    or not isinstance(metadata.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])
                    or type(metadata.get("size_bytes")) is not int
                    or not 0 < metadata["size_bytes"] <= self.MAX_BYTES
                ):
                    raise ValueError(f"Invalid approved digest/size metadata: {name}")
            if (
                sum(metadata["size_bytes"] for metadata in entries.values())
                > self.MAX_BYTES
            ):
                raise ValueError("DSS trust manifest total byte limit exceeded")
        expected_jars = {
            "jar/" + self._filename(artifact) for artifact in self.COMMON_JARS
        }
        if {
            name for name in manifest["artifacts"] if name.startswith("jar/")
        } != expected_jars:
            raise ValueError(
                "DSS trust manifest must list exactly the selected versioned JARs"
            )
        native = {
            "Windows": "javaHeclib.dll",
            "Linux": "libjavaHeclib.so",
            "Darwin": "libjavaHeclib.dylib",
        }.get(platform.system())
        if native is None or "lib/" + native not in manifest["artifacts"]:
            raise ValueError(
                "DSS trust manifest is missing the platform native library"
            )
        if set(manifest["downloads"]) != set(self._download_sources()):
            raise ValueError(
                "DSS download manifest must match the selected artifact versions"
            )
        for name in expected_jars:
            if manifest["downloads"][name] != manifest["artifacts"][name]:
                raise ValueError("JAR runtime and download identities must match")
        return manifest

    def _path(self, name):
        self._safe_name(name)
        path = self.cache_dir / name
        if not path.resolve().is_relative_to(self.cache_dir.resolve()):
            raise ValueError(f"Executable cache path escapes through a link: {name}")
        for candidate in (path, *path.parents):
            if candidate == self.cache_dir.parent:
                break
            if self._is_link(candidate):
                raise ValueError(f"Executable cache links are prohibited: {name}")
        return path

    @staticmethod
    def _verify(path, metadata):
        if not path.is_file() or path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"Executable artifact missing or wrong size: {path.name}")
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            total = 0
            while chunk := stream.read(
                min(1024 * 1024, metadata["size_bytes"] - total + 1)
            ):
                total += len(chunk)
                if total > metadata["size_bytes"]:
                    raise ValueError(
                        f"Executable artifact grew beyond approved size: {path.name}"
                    )
                digest.update(chunk)
        if digest.hexdigest() != metadata["sha256"]:
            raise ValueError(f"Executable artifact checksum mismatch: {path.name}")

    def _unexpected(self, manifest):
        expected = set(manifest["artifacts"]) | set(manifest["downloads"])
        for directory in (self.jar_dir, self.lib_dir):
            if directory.exists():
                for path in directory.rglob("*"):
                    name = path.relative_to(self.cache_dir).as_posix()
                    self._path(name)
                    if (
                        path.is_file()
                        and (
                            path.suffix.lower()
                            in {".jar", ".dll", ".so", ".dylib", ".exe"}
                            or ".so." in path.name.lower()
                        )
                        and name not in expected
                    ):
                        raise ValueError(
                            f"Unexpected executable cache artifact: {name}"
                        )

    def is_installed(self) -> bool:
        """Validate every available artifact; return False only for absent files."""
        manifest = self._trust()
        self._unexpected(manifest)
        complete = True
        for name, metadata in manifest["artifacts"].items():
            path = self._path(name)
            if not path.exists():
                complete = False
            else:
                self._verify(path, metadata)
        for name, metadata in manifest["downloads"].items():
            if name not in manifest["artifacts"]:
                path = self._path(name)
                if path.exists():
                    self._verify(path, metadata)
        return complete

    def get_download_url(self, artifact: dict, source: str = "nexus") -> str:
        """Return the selected package-owned source URL."""
        base = self.MAVEN_BASE if source == "maven" else self.NEXUS_BASE
        return f"{base}/{artifact['group'].replace('.', '/')}/{artifact['artifactId']}/{artifact['version']}/{self._filename(artifact)}"

    def _download_sources(self):
        sources = {
            "jar/" + self._filename(artifact): self.get_download_url(
                artifact, artifact.get("source", "nexus")
            )
            for artifact in self.COMMON_JARS
        }
        artifact = self.NATIVE_LIBS.get(platform.system())
        if artifact:
            sources["lib/" + self._filename(artifact)] = self.get_download_url(artifact)
        return sources

    def _promote(self, temporary, destination, metadata, *, deadline=None):
        self._verify(temporary, metadata)
        self._path(destination.relative_to(self.cache_dir).as_posix())
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Executable download exceeded total transfer deadline")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            self._verify(destination, metadata)

    def download_file(self, url: str, dest: Path, description: str = "") -> Path:
        """Verify cached bytes or atomically admit a bounded approved download."""
        manifest = self._trust()
        name = Path(dest).absolute().relative_to(self.cache_dir).as_posix()
        destination = self._path(name)
        if self._download_sources().get(name) != url:
            raise ValueError(
                "Download source/artifact is not in the selected approved runtime"
            )
        metadata = manifest["downloads"][name]
        if destination.exists():
            self._verify(destination, metadata)
            logger.debug("Using cached HEC Monolith file: %s", destination.name)
            return destination
        if not self.allow_downloads:
            raise RuntimeError(
                f"Sealed DSS runtime is missing {name}; downloads are disabled"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".download-", dir=destination.parent
        )
        temporary = Path(temporary_name)
        deadline = time.monotonic() + self.DOWNLOAD_SECONDS
        logger.debug(
            "Downloading HEC Monolith file: %s", description or destination.name
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                with requests.get(
                    url, stream=True, timeout=(5, 1), allow_redirects=False
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise ValueError(
                            "Executable download redirects/partial responses are prohibited"
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None and int(declared) != metadata["size_bytes"]:
                        raise ValueError(
                            "Executable download Content-Length differs from approved size"
                        )
                    total = 0
                    if not callable(getattr(response.raw, "read1", None)):
                        raise RuntimeError(  # noqa: TRY004 - missing transport capability, not a caller argument type
                            "Verified DSS acquisition requires urllib3 HTTPResponse.read1 support"
                        )
                    # read1 yields available bytes; a trickle cannot hide the
                    # total deadline behind filling a larger requested chunk.
                    with tqdm(
                        total=metadata["size_bytes"],
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                        disable=not self._should_show_progress(),
                    ) as progress:
                        while True:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "Executable download exceeded total transfer deadline"
                                )
                            chunk = response.raw.read1(
                                min(65536, metadata["size_bytes"] - total + 1)
                            )
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "Executable download exceeded total transfer deadline"
                                )
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > metadata["size_bytes"]:
                                raise ValueError(
                                    "Executable download exceeds approved byte limit"
                                )
                            if output.write(chunk) != len(chunk):
                                raise OSError("Incomplete executable staging write")
                            progress.update(len(chunk))
                output.flush()
                os.fsync(output.fileno())
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Executable download exceeded total transfer deadline"
                )
            self._promote(temporary, destination, metadata, deadline=deadline)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    def verify_sha1(
        self, filepath: str | Path, expected_sha1: str | None = None
    ) -> bool:
        """Compare an explicit legacy checksum; SHA-256 manifest trust still governs loading."""
        if not isinstance(expected_sha1, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", expected_sha1
        ):
            return False
        digest = hashlib.sha1()
        with Path(filepath).open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest() == expected_sha1.lower()

    def download_jars(self):
        """Acquire only selected, digest-verified JARs."""
        for artifact in self.COMMON_JARS:
            self.download_file(
                self.get_download_url(artifact, artifact.get("source", "nexus")),
                self.jar_dir / self._filename(artifact),
            )

    def download_native_library(self):
        """Extract only approved native members with bounded size."""
        manifest = self._trust()
        artifact = self.NATIVE_LIBS[platform.system()]
        archive = self.download_file(
            self.get_download_url(artifact), self.lib_dir / self._filename(artifact)
        )
        expected = {
            name[4:]: metadata
            for name, metadata in manifest["artifacts"].items()
            if name.startswith("lib/")
        }
        with zipfile.ZipFile(archive) as bundle:
            members = [member for member in bundle.infolist() if not member.is_dir()]
            if len(bundle.infolist()) > self.MAX_ARCHIVE_MEMBERS:
                raise ValueError("Native archive member limit exceeded")
            all_names = set()
            for member in bundle.infolist():
                member_name = (
                    member.filename[:-1] if member.is_dir() else member.filename
                )
                self._safe_name(member_name)
                if member_name.casefold() in all_names:
                    raise ValueError("Native archive contains colliding members")
                all_names.add(member_name.casefold())
                kind = stat.S_IFMT(member.external_attr >> 16)
                allowed_kind = stat.S_IFDIR if member.is_dir() else stat.S_IFREG
                if kind not in (0, allowed_kind) or member.flag_bits & 1:
                    raise ValueError(
                        "Linked, special or encrypted native archive members are prohibited"
                    )
                if member.is_dir() and not any(
                    name.startswith(member.filename) for name in expected
                ):
                    raise ValueError(
                        "Native archive directory differs from the approved allowlist"
                    )
            names = [member.filename for member in members]
            if len({name.casefold() for name in names}) != len(names) or set(
                names
            ) != set(expected):
                raise ValueError(
                    "Native archive members differ from the approved allowlist"
                )
            for member in members:
                self._safe_name(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                    raise ValueError(
                        "Linked or encrypted native archive members are prohibited"
                    )
                if member.file_size != expected[member.filename]["size_bytes"]:
                    raise ValueError(
                        "Native archive expansion differs from approved byte limit"
                    )
            if sum(member.file_size for member in members) > self.MAX_BYTES:
                raise ValueError("Native archive total expansion limit exceeded")
            for member in members:
                destination = self._path("lib/" + member.filename)
                metadata = expected[member.filename]
                if destination.exists():
                    self._verify(destination, metadata)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".extract-", dir=destination.parent
                )
                temporary = Path(temporary_name)
                try:
                    with (
                        os.fdopen(descriptor, "wb") as output,
                        bundle.open(member) as source,
                    ):
                        total = 0
                        while chunk := source.read(65536):
                            total += len(chunk)
                            if total > metadata["size_bytes"]:
                                raise ValueError(
                                    "Native archive expansion exceeded approved limit"
                                )
                            if output.write(chunk) != len(chunk):
                                raise OSError("Incomplete executable extraction write")
                        output.flush()
                        os.fsync(output.fileno())
                    self._promote(temporary, destination, metadata)
                finally:
                    temporary.unlink(missing_ok=True)

    def install(self, force: bool = False):
        """Verify or complete the cache; force never overwrites conflicting bytes."""
        if self.is_installed() and not force:
            logger.debug("HEC Monolith already installed")
            return
        logger.debug("Installing HEC Monolith libraries in: %s", self.cache_dir)
        self.download_jars()
        self.download_native_library()
        if not self.is_installed():
            raise RuntimeError("DSS runtime remains incomplete after acquisition")
        logger.debug("HEC Monolith installation complete")

    def get_classpath(self) -> list[str]:
        """Return only exact, currently verified approved JAR paths."""
        if not self.is_installed():
            raise RuntimeError(
                "Verified DSS runtime is incomplete; provision the sealed cache first"
            )
        return [
            str(self._path("jar/" + self._filename(artifact)))
            for artifact in self.COMMON_JARS
        ]

    def get_library_path(self) -> str:
        """Return the native library directory after full integrity validation."""
        if not self.is_installed():
            raise RuntimeError(
                "Verified DSS runtime is incomplete; provision the sealed cache first"
            )
        return str(self.lib_dir)

    def get_info(self) -> dict:
        """Return a validated inventory without discovering new trust."""
        manifest = self._trust()
        jars = [name for name in manifest["artifacts"] if name.startswith("jar/")]
        libs = [name for name in manifest["artifacts"] if name.startswith("lib/")]
        return {
            "trust_policy": self.TRUST_POLICY,
            "installed": self.is_installed(),
            "cache_dir": str(self.cache_dir),
            "num_jars": len(jars),
            "jar_files": [Path(name).name for name in jars],
            "lib_files": [Path(name).name for name in libs],
            "total_size_mb": round(
                sum(value["size_bytes"] for value in manifest["artifacts"].values())
                / 1024**2,
                2,
            ),
            "platform": platform.system(),
            "downloads_allowed": self.allow_downloads,
        }
