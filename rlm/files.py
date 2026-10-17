"""
Lazy file loading for RLM contexts.

Instead of a single context string, an RLM can be given a collection of files
(local paths, glob patterns, or gs:// URIs). Files are exposed in the REPL as
LazyFile objects with progressive disclosure, so the model decides what to load:

- Level 0 (free):      name, path, extension            - from the listing
- Level 1 (cheap):     size, modified, mime_type        - stat / metadata request
- Level 2 (expensive): content, pages, read_bytes()     - full read + parse

Content is parsed and cached on first access; a LazyFile can be shared by
parallel child RLMs (loading is thread-safe).

Inspired by the lazy file module of ADK-RLM
(github.com/LiamConnell/adk-python, contributing/samples/rlm).
"""

import csv
import fnmatch
import glob as globlib
import io
import json
import mimetypes
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union


# =============================================================================
# Data types
# =============================================================================

@dataclass
class FileMetadata:
    """Metadata about a file (Level 1)."""

    name: str
    path: str
    source_type: str
    size_bytes: int
    mime_type: Optional[str] = None
    last_modified: Optional[datetime] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedContent:
    """Text extracted from a file (Level 2)."""

    text: str
    pages: Optional[List[str]] = None       # e.g. PDF pages
    metadata: Dict[str, Any] = field(default_factory=dict)


def _name_of(path: str) -> str:
    return path.rstrip("/").split("/")[-1].split("\\")[-1]


def _extension_of(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _has_glob(pattern: str) -> bool:
    return any(c in pattern for c in "*?[")


# =============================================================================
# Sources
# =============================================================================

class FileSource(ABC):
    """
    Where files come from (local disk, cloud storage, ...).

    Sources only read paths they resolved themselves, so code running in the
    REPL cannot use a source object to read arbitrary files.
    """

    source_type: str = "base"

    def _allow(self, paths: List[str]) -> List[str]:
        if not hasattr(self, "_allowed"):
            self._allowed: Set[str] = set()
        self._allowed.update(paths)
        return paths

    def _check(self, path: str) -> str:
        if path not in getattr(self, "_allowed", set()):
            raise PermissionError(f"Path was not resolved by this source: {path}")
        return path

    @abstractmethod
    def resolve(self, pattern: str) -> List[str]:
        """Resolve a path or glob pattern to concrete file paths."""

    @abstractmethod
    def get_metadata(self, path: str) -> FileMetadata:
        """Get metadata without reading content."""

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read the full file."""


class LocalFileSource(FileSource):
    """Files on the local filesystem; relative paths resolve against base_path."""

    source_type = "local"

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path).resolve() if base_path else Path.cwd()

    def _full(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        return p if p.is_absolute() else self.base_path / p

    def resolve(self, pattern: str) -> List[str]:
        full = self._full(pattern)
        if _has_glob(str(full)):
            found = sorted(m for m in globlib.glob(str(full), recursive=True) if os.path.isfile(m))
        elif full.is_dir():
            found = sorted(str(p) for p in full.rglob("*") if p.is_file())
        else:
            found = [str(full)] if full.is_file() else []
        return self._allow(found)

    def get_metadata(self, path: str) -> FileMetadata:
        p = Path(self._check(path))
        st = p.stat()
        return FileMetadata(
            name=p.name,
            path=str(p),
            source_type=self.source_type,
            size_bytes=st.st_size,
            mime_type=mimetypes.guess_type(str(p))[0],
            last_modified=datetime.fromtimestamp(st.st_mtime),
        )

    def read_bytes(self, path: str) -> bytes:
        return Path(self._check(path)).read_bytes()


class GCSFileSource(FileSource):
    """
    Files in a Google Cloud Storage bucket, addressed as gs://bucket/path.

    Requires `pip install google-cloud-storage` and Application Default
    Credentials (or a service-account JSON via credentials_path).
    Glob patterns are matched against object names under the longest
    non-glob prefix (e.g. gs://bucket/reports/**/*.pdf).
    """

    source_type = "gcs"

    def __init__(self, client: Any = None, project: Optional[str] = None,
                 credentials_path: Optional[str] = None):
        if client is None:
            try:
                from google.cloud import storage
            except ImportError:
                raise ImportError(
                    "GCS support requires google-cloud-storage. "
                    "Install with: pip install google-cloud-storage"
                )
            if credentials_path:
                client = storage.Client.from_service_account_json(credentials_path, project=project)
            else:
                client = storage.Client(project=project)
        self.client = client
        self._blobs: Dict[str, Any] = {}  # listing results carry metadata for free

    @staticmethod
    def _split(uri: str) -> Tuple[str, str]:
        if not uri.startswith("gs://"):
            raise ValueError(f"Not a gs:// URI: {uri}")
        bucket, _, name = uri[len("gs://"):].partition("/")
        return bucket, name

    def resolve(self, pattern: str) -> List[str]:
        bucket, name = self._split(pattern)
        if not _has_glob(name):
            blob = self.client.bucket(bucket).get_blob(name) if name else None
            if blob is not None:
                self._blobs[pattern] = blob
                return self._allow([pattern])
            # Treat as a "folder" prefix
            name = name.rstrip("/") + "/**" if name else "**"
        prefix = name
        for i, ch in enumerate(name):
            if ch in "*?[":
                prefix = name[:i]
                break
        prefix = prefix[: prefix.rfind("/") + 1] if "/" in prefix else ""
        # "**/" also matches zero directories
        alt = name.replace("**/", "")
        matches = []
        for blob in self.client.list_blobs(bucket, prefix=prefix or None):
            if blob.name.endswith("/"):
                continue
            if fnmatch.fnmatch(blob.name, name) or fnmatch.fnmatch(blob.name, alt):
                uri = f"gs://{bucket}/{blob.name}"
                self._blobs[uri] = blob
                matches.append(uri)
        return self._allow(sorted(matches))

    def _blob(self, uri: str):
        self._check(uri)
        blob = self._blobs.get(uri)
        if blob is None:
            bucket, name = self._split(uri)
            blob = self.client.bucket(bucket).get_blob(name)
            if blob is None:
                raise FileNotFoundError(uri)
            self._blobs[uri] = blob
        return blob

    def get_metadata(self, path: str) -> FileMetadata:
        blob = self._blob(path)
        return FileMetadata(
            name=_name_of(blob.name),
            path=path,
            source_type=self.source_type,
            size_bytes=blob.size or 0,
            mime_type=blob.content_type or mimetypes.guess_type(blob.name)[0],
            last_modified=blob.updated,
            extra={"bucket": blob.bucket.name if getattr(blob, "bucket", None) else None},
        )

    def read_bytes(self, path: str) -> bytes:
        return self._blob(path).download_as_bytes()


# =============================================================================
# Parsers
# =============================================================================

class FileParser(ABC):
    """Turns file bytes into text."""

    extensions: Sequence[str] = ()

    def can_parse(self, name: str) -> bool:
        return _extension_of(name) in self.extensions

    @abstractmethod
    def parse(self, data: bytes, name: str) -> ParsedContent:
        """Parse raw bytes."""


class TextParser(FileParser):
    """Plain text, markup, code, JSON (pretty-printed) and CSV (with row counts)."""

    extensions = (
        ".txt", ".md", ".markdown", ".rst", ".log", ".json", ".jsonl", ".csv", ".tsv",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html", ".htm",
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".sql", ".sh",
    )

    def parse(self, data: bytes, name: str) -> ParsedContent:
        text = data.decode("utf-8", errors="replace")
        ext = _extension_of(name)
        meta: Dict[str, Any] = {"format": ext.lstrip(".") or "text"}
        if ext == ".json":
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            except ValueError:
                pass
        elif ext in (".csv", ".tsv"):
            try:
                rows = list(csv.reader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ","))
                meta["rows"] = max(0, len(rows) - 1)
                meta["columns"] = rows[0] if rows else []
            except csv.Error:
                pass
        meta["lines"] = text.count("\n") + 1
        return ParsedContent(text=text, metadata=meta)


class PDFParser(FileParser):
    """PDF text per page. Requires `pip install pypdf`."""

    extensions = (".pdf",)

    def parse(self, data: bytes, name: str) -> ParsedContent:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("PDF support requires pypdf. Install with: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = "\n\n".join(f"[Page {i + 1}]\n{p}" for i, p in enumerate(pages))
        return ParsedContent(text=text, pages=pages, metadata={"format": "pdf", "page_count": len(pages)})


# =============================================================================
# Lazy files
# =============================================================================

class LazyFile:
    """
    A file reference whose metadata and content load on first access.

    Read-only from REPL code: attributes cannot be reassigned, so generated
    code cannot point a LazyFile at a different path.
    """

    __slots__ = ("_path", "_source", "_parser", "_lock", "_metadata", "_raw", "_parsed", "_frozen")

    def __init__(self, path: str, source: FileSource, parser: Optional[FileParser]):
        self._path = path
        self._source = source
        self._parser = parser
        self._lock = threading.Lock()
        self._metadata: Optional[FileMetadata] = None
        self._raw: Optional[bytes] = None
        self._parsed: Optional[ParsedContent] = None
        self._frozen = True

    def __setattr__(self, key, value):
        if getattr(self, "_frozen", False) and key not in ("_metadata", "_raw", "_parsed"):
            raise AttributeError("LazyFile is read-only")
        object.__setattr__(self, key, value)

    # ---- Level 0: no I/O -------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return _name_of(self._path)

    @property
    def extension(self) -> str:
        return _extension_of(self.name)

    @property
    def source_type(self) -> str:
        return self._source.source_type

    @property
    def level(self) -> int:
        """0 = path only, 1 = metadata loaded, 2 = content loaded."""
        if self._parsed is not None or self._raw is not None:
            return 2
        return 1 if self._metadata is not None else 0

    # ---- Level 1: metadata -----------------------------------------------

    @property
    def metadata(self) -> FileMetadata:
        if self._metadata is None:
            with self._lock:
                if self._metadata is None:
                    self._metadata = self._source.get_metadata(self._path)
        return self._metadata

    @property
    def size(self) -> int:
        return self.metadata.size_bytes

    @property
    def size_kb(self) -> float:
        return self.size / 1024

    @property
    def modified(self) -> Optional[datetime]:
        return self.metadata.last_modified

    @property
    def mime_type(self) -> Optional[str]:
        return self.metadata.mime_type

    # ---- Level 2: content ------------------------------------------------

    def read_bytes(self) -> bytes:
        """Raw file bytes (cached)."""
        if self._raw is None:
            with self._lock:
                if self._raw is None:
                    self._raw = self._source.read_bytes(self._path)
        return self._raw

    def _parse(self) -> ParsedContent:
        if self._parsed is None:
            data = self.read_bytes()
            with self._lock:
                if self._parsed is None:
                    if self._parser is not None:
                        self._parsed = self._parser.parse(data, self.name)
                    else:
                        self._parsed = ParsedContent(
                            text=data.decode("utf-8", errors="replace"),
                            metadata={"format": "unknown"},
                        )
        return self._parsed

    @property
    def content(self) -> str:
        """Parsed text content (triggers read + parse, cached)."""
        return self._parse().text

    @property
    def pages(self) -> Optional[List[str]]:
        """Per-page text for paged formats such as PDF, else None."""
        return self._parse().pages

    @property
    def parsed_metadata(self) -> Dict[str, Any]:
        """Parser details, e.g. page_count, rows, columns, lines."""
        return self._parse().metadata

    def __repr__(self) -> str:
        return f"<LazyFile {self.name!r} level={self.level}>"

    __str__ = __repr__


class LazyFileCollection:
    """A list of LazyFiles with filters that need no I/O and batch helpers."""

    def __init__(self, files: Iterable[LazyFile] = ()):
        self.files: List[LazyFile] = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[LazyFile]:
        return iter(self.files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LazyFileCollection(self.files[idx])
        return self.files[idx]

    def __bool__(self) -> bool:
        return bool(self.files)

    # ---- Level 0 ---------------------------------------------------------

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.files]

    @property
    def paths(self) -> List[str]:
        return [f.path for f in self.files]

    @property
    def extensions(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.files:
            counts[f.extension or "(none)"] = counts.get(f.extension or "(none)", 0) + 1
        return counts

    def by_extension(self, ext: str) -> "LazyFileCollection":
        ext = (ext if ext.startswith(".") else "." + ext).lower()
        return LazyFileCollection(f for f in self.files if f.extension == ext)

    def by_name(self, pattern: str) -> "LazyFileCollection":
        """Filter by glob pattern on the file name, e.g. '*2024*.pdf'."""
        return LazyFileCollection(f for f in self.files if fnmatch.fnmatch(f.name, pattern))

    def search(self, keyword: str) -> "LazyFileCollection":
        """Filter by case-insensitive keyword in the path."""
        k = keyword.lower()
        return LazyFileCollection(f for f in self.files if k in f.path.lower())

    # ---- Level 1 / 2 -----------------------------------------------------

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def loaded_count(self) -> int:
        return sum(1 for f in self.files if f.level == 2)

    def grep(self, pattern: str, max_matches: int = 100, flags: int = 0) -> List[Dict[str, Any]]:
        """Regex-search file contents (loads files). Returns dicts with file, line_no, line."""
        import re
        rx = re.compile(pattern, flags)
        out: List[Dict[str, Any]] = []
        for f in self.files:
            for i, line in enumerate(f.content.splitlines(), 1):
                if rx.search(line):
                    out.append({"file": f.name, "path": f.path, "line_no": i, "line": line[:500]})
                    if len(out) >= max_matches:
                        return out
        return out

    def summary(self, max_names: int = 50) -> str:
        """Short description without loading content."""
        exts = ", ".join(f"{e}: {n}" for e, n in sorted(self.extensions.items()))
        lines = [f"{len(self.files)} files ({exts})"]
        for f in self.files[:max_names]:
            lines.append(f"  - {f.path}")
        if len(self.files) > max_names:
            lines.append(f"  ... and {len(self.files) - max_names} more (see files.names)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<LazyFileCollection {len(self.files)} files, {self.loaded_count} loaded>"

    __str__ = __repr__


# =============================================================================
# Loader
# =============================================================================

class FileLoader:
    """
    Resolves paths, globs and gs:// URIs into a LazyFileCollection.

    Example:
        loader = FileLoader()
        files = loader.create_lazy_files(["./docs/**/*.md", "report.pdf"])
    """

    def __init__(self, sources: Optional[Dict[str, FileSource]] = None,
                 parsers: Optional[List[FileParser]] = None,
                 base_path: Optional[Union[str, Path]] = None):
        self.sources: Dict[str, FileSource] = {"local": LocalFileSource(base_path)}
        self.sources.update(sources or {})
        self.parsers: List[FileParser] = parsers or [TextParser(), PDFParser()]

    def register_source(self, name: str, source: FileSource):
        """Register a source; 'gcs' handles gs:// URIs."""
        self.sources[name] = source

    def register_parser(self, parser: FileParser, first: bool = True):
        if first:
            self.parsers.insert(0, parser)
        else:
            self.parsers.append(parser)

    def _source_for(self, path: str) -> FileSource:
        if path.startswith("gs://"):
            if "gcs" not in self.sources:
                self.sources["gcs"] = GCSFileSource()
            return self.sources["gcs"]
        if "://" in path:
            scheme = path.split("://", 1)[0]
            if scheme in self.sources:
                return self.sources[scheme]
            raise ValueError(f"No file source registered for '{scheme}://' "
                             f"(use loader.register_source('{scheme}', source))")
        return self.sources["local"]

    def _parser_for(self, name: str) -> Optional[FileParser]:
        return next((p for p in self.parsers if p.can_parse(name)), None)

    def create_lazy_files(self, patterns: Union[str, Sequence[Union[str, LazyFile]], LazyFileCollection]
                          ) -> LazyFileCollection:
        """Resolve patterns into lazy files (no content is read)."""
        if isinstance(patterns, LazyFileCollection):
            return patterns
        if isinstance(patterns, (str, LazyFile)):
            patterns = [patterns]
        files: List[LazyFile] = []
        seen = set()
        for pat in patterns:
            if isinstance(pat, LazyFile):
                if pat.path not in seen:
                    seen.add(pat.path)
                    files.append(pat)
                continue
            source = self._source_for(str(pat))
            resolved = source.resolve(str(pat))
            if not resolved:
                raise FileNotFoundError(f"No files match: {pat}")
            for path in resolved:
                if path not in seen:
                    seen.add(path)
                    files.append(LazyFile(path, source, self._parser_for(_name_of(path))))
        return LazyFileCollection(files)
