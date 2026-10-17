"""
Tests for lazy file loading (rlm/files.py) and run(files=...).

No API keys needed. The PDF test needs pypdf (pip install pypdf).
Run with: pytest tests/test_files.py
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rlm import RecursiveLanguageModel
from rlm.files import (
    FileLoader,
    GCSFileSource,
    LazyFile,
    LazyFileCollection,
    LocalFileSource,
)
from tests.test_recursion import ScriptedProvider


@pytest.fixture
def docs(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# Alpha\nThe launch date is 2024-03-01.\n")
    (tmp_path / "notes" / "b.md").write_text("# Beta\nNothing to see.\n")
    (tmp_path / "data.json").write_text('{"k": [1, 2]}')
    (tmp_path / "table.csv").write_text("name,score\nann,3\nbob,5\n")
    (tmp_path / "plain.unknownext").write_text("raw text")
    return tmp_path


class TestLocalSourceAndLoader:

    def test_glob_and_directory_resolution(self, docs):
        loader = FileLoader(base_path=docs)
        files = loader.create_lazy_files(["notes/*.md", "data.json"])
        assert files.names == ["a.md", "b.md", "data.json"]
        assert len(loader.create_lazy_files(str(docs / "notes"))) == 2   # directory = all files
        assert len(loader.create_lazy_files("**/*.md")) == 2

    def test_missing_pattern_raises(self, docs):
        with pytest.raises(FileNotFoundError):
            FileLoader(base_path=docs).create_lazy_files("nope/*.txt")

    def test_duplicates_are_removed(self, docs):
        files = FileLoader(base_path=docs).create_lazy_files(["notes/a.md", "notes/*.md"])
        assert files.names == ["a.md", "b.md"]

    def test_unknown_scheme_raises(self, docs):
        with pytest.raises(ValueError, match="s3://"):
            FileLoader(base_path=docs).create_lazy_files("s3://bucket/x.txt")


class TestLazyFile:

    def test_progressive_levels(self, docs):
        f = FileLoader(base_path=docs).create_lazy_files("notes/a.md")[0]
        assert f.level == 0
        assert f.name == "a.md" and f.extension == ".md"
        assert f.level == 0                      # level-0 properties do no I/O
        assert f.size > 0
        assert f.level == 1
        assert "2024-03-01" in f.content
        assert f.level == 2
        assert f.parsed_metadata["lines"] == 3

    def test_read_only(self, docs):
        f = FileLoader(base_path=docs).create_lazy_files("notes/a.md")[0]
        with pytest.raises(AttributeError):
            f._path = "/etc/passwd"

    def test_source_refuses_unresolved_paths(self, docs):
        source = LocalFileSource(docs)
        source.resolve("notes/a.md")
        with pytest.raises(PermissionError):
            source.read_bytes(str(docs / "notes" / "b.md"))
        with pytest.raises(PermissionError):
            source.read_bytes("/etc/passwd")

    def test_parsers(self, docs):
        files = FileLoader(base_path=docs).create_lazy_files(["data.json", "table.csv", "plain.unknownext"])
        js, csv_file, plain = files
        assert '"k": [\n' in js.content                         # pretty-printed
        assert csv_file.parsed_metadata["rows"] == 2
        assert csv_file.parsed_metadata["columns"] == ["name", "score"]
        assert plain.content == "raw text"                      # no parser: decoded text

    def test_pdf_pages(self):
        pytest.importorskip("pypdf")
        pdf = os.path.join(os.path.dirname(__file__), "Attention_is_all_you_need.pdf")
        if not os.path.exists(pdf):
            pytest.skip("sample PDF not present")
        f = FileLoader().create_lazy_files(pdf)[0]
        assert f.parsed_metadata["page_count"] == len(f.pages) > 1
        assert "[Page 1]" in f.content and "attention" in f.content.lower()


class TestCollection:

    def test_filters_without_io(self, docs):
        files = FileLoader(base_path=docs).create_lazy_files("**/*")
        assert files.by_extension("md").names == ["a.md", "b.md"]
        assert files.by_name("dat*").names == ["data.json"]
        assert files.search("NOTES").names == ["a.md", "b.md"]
        assert files.extensions[".md"] == 2
        assert files.loaded_count == 0
        assert isinstance(files[:2], LazyFileCollection)

    def test_grep_loads_and_finds(self, docs):
        files = FileLoader(base_path=docs).create_lazy_files("notes/*.md")
        hits = files.grep(r"\d{4}-\d{2}-\d{2}")
        assert [(h["file"], h["line_no"]) for h in hits] == [("a.md", 2)]
        assert files.loaded_count == 2

    def test_summary_lists_paths(self, docs):
        files = FileLoader(base_path=docs).create_lazy_files("notes/*.md")
        s = files.summary()
        assert "2 files" in s and "a.md" in s


class TestGCSSource:

    def _client(self):
        def blob(name, data):
            b = MagicMock()
            b.name = name
            b.size = len(data)
            b.content_type = "text/plain"
            b.updated = None
            b.bucket = SimpleNamespace(name="bkt")
            b.download_as_bytes.return_value = data
            return b

        blobs = [blob("reports/2024/q1.txt", b"q1 revenue up"),
                 blob("reports/q0.txt", b"q0"),
                 blob("reports/2024/img.png", b"\x89PNG"),
                 blob("other/x.txt", b"x")]
        client = MagicMock()
        client.list_blobs.side_effect = lambda bucket, prefix=None: [
            b for b in blobs if prefix is None or b.name.startswith(prefix)]
        client.bucket.return_value.get_blob.side_effect = lambda name: next(
            (b for b in blobs if b.name == name), None)
        return client

    def test_glob_listing_and_lazy_download(self):
        client = self._client()
        loader = FileLoader(sources={"gcs": GCSFileSource(client=client)})
        files = loader.create_lazy_files("gs://bkt/reports/**/*.txt")
        assert files.paths == ["gs://bkt/reports/2024/q1.txt", "gs://bkt/reports/q0.txt"]
        f = files[0]
        assert f.size == len(b"q1 revenue up")          # from listing, no download
        f.metadata
        assert not client.bucket.return_value.get_blob.called
        assert f.content == "q1 revenue up"

    def test_single_object_and_unresolved_refused(self):
        source = GCSFileSource(client=self._client())
        assert source.resolve("gs://bkt/other/x.txt") == ["gs://bkt/other/x.txt"]
        with pytest.raises(PermissionError):
            source.read_bytes("gs://bkt/reports/q0.txt")


class TestRunWithFiles:

    def test_files_exposed_in_repl_and_prompt(self, docs):
        provider = ScriptedProvider(
            scripts={0: "hits = files.grep(r'launch date is (\\S+)')\nFINAL(hits[0]['line'])"},
            context_window=1_000_000,
        )
        rlm = RecursiveLanguageModel(provider=provider, log_level="ERROR", enable_cache=False,
                                     file_loader=FileLoader(base_path=docs))
        result = rlm.run("When is the launch?", files=["notes/*.md"], verbose=False)

        assert result == "The launch date is 2024-03-01."   # REPL mode even though tiny
        system = rlm.history[0]["content"]
        assert "## Files:" in system and "2 files" in system and "a.md" in system

    def test_rlm_query_hands_file_to_child(self, docs):
        provider = ScriptedProvider(scripts={
            0: "FINAL(rlm_query_batched(['count lines'] * len(files), contexts=list(files)))",
            1: "FINAL(f'{files[0].name}:{len(files[0].content.splitlines())}')",
        }, context_window=1_000_000)
        rlm = RecursiveLanguageModel(provider=provider, log_level="ERROR", enable_cache=False,
                                     file_loader=FileLoader(base_path=docs), max_depth=1)
        result = rlm.run("t", files="notes/*.md", verbose=False)
        assert result == ["a.md:2", "b.md:2"]

    def test_max_depth_inlines_file_content(self, docs):
        provider = ScriptedProvider(
            scripts={0: "FINAL(rlm_query('summarize', context=files[0]))"},
            sub_call_fn=lambda prompt: "saw date" if "2024-03-01" in prompt else "no date",
            context_window=1_000_000,
        )
        rlm = RecursiveLanguageModel(provider=provider, log_level="ERROR", enable_cache=False,
                                     file_loader=FileLoader(base_path=docs), max_depth=0)
        assert rlm.run("t", files="notes/a.md", verbose=False) == "saw date"
