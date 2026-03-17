"""Tests for benchmarks.download module.

Tests file download utilities, HuggingFace dataset fetching, and
download orchestration using mocked HTTP requests.
"""

from unittest.mock import MagicMock, patch

from benchmarks.download import (
    DOWNLOADERS,
    download_all,
    download_file,
    download_hf_dataset,
)

# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Tests for download_file."""

    def test_successful_download(self, tmp_path):
        """Downloads file content to destination path."""
        dest = tmp_path / "output.txt"
        content = b"Hello, world!"

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(content))}
        mock_resp.iter_content.return_value = [content]
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_file("http://example.com/file.txt", dest, desc="test")

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == content

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if they don't exist."""
        dest = tmp_path / "a" / "b" / "output.txt"

        mock_resp = MagicMock()
        mock_resp.headers = {}
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_file("http://example.com/f", dest)

        assert result is True
        assert dest.exists()

    def test_http_error_returns_false(self, tmp_path):
        """Returns False on HTTP errors."""
        dest = tmp_path / "output.txt"

        with patch("benchmarks.download.requests.get", side_effect=Exception("Connection failed")):
            result = download_file("http://bad-url.com/file", dest)

        assert result is False
        assert not dest.exists()

    def test_chunked_download(self, tmp_path):
        """Handles chunked downloads correctly."""
        dest = tmp_path / "output.bin"
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(sum(len(c) for c in chunks))}
        mock_resp.iter_content.return_value = chunks
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_file("http://example.com/large", dest)

        assert result is True
        assert dest.read_bytes() == b"chunk1chunk2chunk3"


# ---------------------------------------------------------------------------
# download_hf_dataset
# ---------------------------------------------------------------------------


class TestDownloadHFDataset:
    """Tests for download_hf_dataset."""

    def test_single_page_download(self, tmp_path):
        """Downloads a small dataset in a single page."""
        dest = tmp_path / "output.jsonl"

        rows = [{"row": {"claim": f"claim_{i}", "label": "SUPPORTS"}} for i in range(5)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rows": rows}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_hf_dataset("test/dataset", "config", "train", dest)

        assert result is True
        assert dest.exists()
        lines = dest.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_empty_response_returns_false(self, tmp_path):
        """Returns False when no data is fetched."""
        dest = tmp_path / "output.jsonl"

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rows": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_hf_dataset("test/dataset", "", "train", dest)

        assert result is False

    def test_max_rows_limit(self, tmp_path):
        """Respects max_rows limit."""
        dest = tmp_path / "output.jsonl"

        # Return 100 rows per page
        rows = [{"row": {"id": i}} for i in range(100)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rows": rows}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_hf_dataset("test/dataset", "", "train", dest, max_rows=50)

        assert result is True
        lines = dest.read_text().strip().split("\n")
        assert len(lines) == 50

    def test_api_error_returns_partial(self, tmp_path):
        """API errors mid-download still save partial results."""
        dest = tmp_path / "output.jsonl"

        first_response = MagicMock()
        first_response.json.return_value = {"rows": [{"row": {"id": 1}}]}
        first_response.raise_for_status = MagicMock()

        error_response = MagicMock()
        error_response.raise_for_status.side_effect = Exception("API Error")

        with patch("benchmarks.download.requests.get", side_effect=[first_response, error_response]):
            # Should save partial results from first page
            result = download_hf_dataset("test/dataset", "", "train", dest)

        # First page had data, so it should be saved
        assert result is True
        assert dest.exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories for output file."""
        dest = tmp_path / "sub" / "dir" / "output.jsonl"

        rows = [{"row": {"id": 1}}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rows": rows}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmarks.download.requests.get", return_value=mock_resp):
            result = download_hf_dataset("test", "", "train", dest)

        assert result is True
        assert dest.parent.exists()


# ---------------------------------------------------------------------------
# DOWNLOADERS registry
# ---------------------------------------------------------------------------


class TestDownloaders:
    """Tests for DOWNLOADERS registry."""

    def test_all_benchmarks_registered(self):
        assert "scifact" in DOWNLOADERS
        assert "fever" in DOWNLOADERS
        assert "qasper" in DOWNLOADERS
        assert "hagrid" in DOWNLOADERS

    def test_all_downloaders_are_callable(self):
        for name, fn in DOWNLOADERS.items():
            assert callable(fn), f"{name} downloader is not callable"


# ---------------------------------------------------------------------------
# download_all
# ---------------------------------------------------------------------------


class TestDownloadAll:
    """Tests for download_all."""

    def test_calls_all_downloaders(self):
        """Calls all registered downloaders."""
        with patch.dict(
            "benchmarks.download.DOWNLOADERS",
            {"a": MagicMock(return_value=True), "b": MagicMock(return_value=True)},
            clear=True,
        ):
            results = download_all()
            assert results == {"a": True, "b": True}

    def test_returns_failure_results(self):
        """Reports failure for individual downloaders."""
        with patch.dict(
            "benchmarks.download.DOWNLOADERS",
            {"ok": MagicMock(return_value=True), "fail": MagicMock(return_value=False)},
            clear=True,
        ):
            results = download_all()
            assert results["ok"] is True
            assert results["fail"] is False

    def test_include_wiki_flag(self):
        """Includes wiki download when flag is set."""
        with (
            patch.dict("benchmarks.download.DOWNLOADERS", {}),
            patch("benchmarks.download.download_fever_wiki", return_value=True) as mock_wiki,
        ):
            results = download_all(include_wiki=True)
            mock_wiki.assert_called_once()
            assert results["fever_wiki"] is True

    def test_exclude_wiki_by_default(self):
        """Wiki download is excluded by default."""
        with (
            patch.dict("benchmarks.download.DOWNLOADERS", {}),
            patch("benchmarks.download.download_fever_wiki") as mock_wiki,
        ):
            download_all(include_wiki=False)
            mock_wiki.assert_not_called()
