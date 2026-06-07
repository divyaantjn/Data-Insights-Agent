"""
test_s3_utility.py - Comprehensive unit tests for S3Utility and get_s3_key_from_url.

Coverage targets:
- get_s3_key_from_url        : https, s3://, unknown scheme, URL encoding, + replacement
- S3Utility.__init__         : env-var wiring, region priority
- upload_file                : success, ClientError
- upload_file_by_url         : success, ClientError
- get_data_from_s3_by_url    : success, NoSuchKey, AccessDenied, other ClientError
- get_file                   : success, NoSuchKey, other ClientError
- _get_s3_object             : success, ClientError
- generate_presigned_url     : success, NoSuchKey, other ClientError, default expiry
- delete_file                : success, NoSuchKey, other ClientError
- delete_file_by_url         : success, NoSuchKey, other ClientError
- create_zip_and_upload_for_urls : success, filenames inside zip
- extract_filename_from_s3_url    : UUID suffix, plain name, https url, encoded chars
- copy_s3_file_to_new_path        : virtual-hosted, s3://, spaces, exception, copy args
- _parse_virtual_hosted_style     : success, empty key raises
- _parse_path_style              : success, empty key raises
- _parse_s3_url_for_copy          : all URL variants + error paths
"""

import io
import os
import sys
import zipfile
from unittest.mock import MagicMock, patch, call
from urllib.parse import urlparse

import pytest
from botocore.exceptions import ClientError

def _install_fastapi_stub():
    """
    Insert a minimal fastapi stub into sys.modules so s3_utility can be
    imported even when fastapi is not installed.  Returns the stub HTTPException
    class and a flag indicating whether the stub was freshly created.
    """
    class _StubHTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = ""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

        def __repr__(self):
            return (
                f"HTTPException(status_code={self.status_code},"
                f" detail={self.detail!r})"
            )

    stub_fastapi           = MagicMock()
    stub_fastapi.HTTPException = _StubHTTPException
    stub_fastapi.exceptions    = MagicMock()
    stub_fastapi.exceptions.HTTPException = _StubHTTPException

    sys.modules.setdefault("fastapi",            stub_fastapi)
    sys.modules.setdefault("fastapi.exceptions", stub_fastapi.exceptions)

    return _StubHTTPException


# Install stub (no-op if fastapi is already present) BEFORE first import.
_StubHTTPException = _install_fastapi_stub()

# ---------------------------------------------------------------------------
# Import the module under test.
# ---------------------------------------------------------------------------
from src.utils.s3_utility import S3Utility, get_s3_key_from_url  # noqa: E402

# ---------------------------------------------------------------------------
# Resolve the HTTPException class the module ACTUALLY uses at runtime.
#
# The module does something like:
#   from fastapi import HTTPException   (or from fastapi.exceptions import ...)
#
# After import, inspect the module's own globals to find whichever class it
# bound to the name "HTTPException".  Fall back to our stub if not found.
# ---------------------------------------------------------------------------
import src.utils.s3_utility as _s3_mod

_RuntimeHTTPException = getattr(_s3_mod, "HTTPException", None)
if _RuntimeHTTPException is None:
    # Try the fastapi module that was active during import
    _fa = sys.modules.get("fastapi")
    _RuntimeHTTPException = getattr(_fa, "HTTPException", _StubHTTPException)

# This is the class we must pass to pytest.raises() everywhere.
HTTPException = _RuntimeHTTPException


# ---------------------------------------------------------------------------
# 2. Helper – build a botocore ClientError
# ---------------------------------------------------------------------------
def make_client_error(code: str, message: str = "msg") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="op",
    )


# ---------------------------------------------------------------------------
# 3. Shared fixture — boto3 is patched at the module level of s3_utility
# ---------------------------------------------------------------------------
@pytest.fixture
def s3util(monkeypatch):
    """S3Utility instance with a fully-mocked boto3 client and known bucket."""
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("AWS_REGION",     "us-east-1")

    with patch("src.utils.s3_utility.boto3") as mock_boto_module:
        mock_client = MagicMock()
        mock_boto_module.client.return_value = mock_client

        util = S3Utility()
        # Ensure every test operates on the same mock object
        util.s3_client = mock_client

        yield util


# ===========================================================================
# Section 1 – get_s3_key_from_url (pure function, no boto3 dependency)
# ===========================================================================
class TestGetS3KeyFromUrl:
    def test_https_virtual_hosted_returns_key(self):
        url = "https://my-bucket.s3.amazonaws.com/folder/file.txt"
        assert get_s3_key_from_url(url) == "folder/file.txt"

    def test_https_url_decodes_percent_encoding(self):
        url = "https://my-bucket.s3.amazonaws.com/folder/file%20name.txt"
        assert get_s3_key_from_url(url) == "folder/file name.txt"

    def test_https_url_replaces_plus_with_space(self):
        url = "https://my-bucket.s3.amazonaws.com/folder/file+name.txt"
        assert get_s3_key_from_url(url) == "folder/file name.txt"

    def test_s3_scheme_returns_key(self):
        url = "s3://my-bucket/folder/file.txt"
        assert get_s3_key_from_url(url) == "folder/file.txt"

    def test_s3_scheme_no_key_returns_empty(self):
        assert get_s3_key_from_url("s3://my-bucket") == ""

    def test_s3_scheme_deep_path(self):
        assert get_s3_key_from_url("s3://my-bucket/a/b/c/d.csv") == "a/b/c/d.csv"

    def test_unknown_scheme_returns_empty(self):
        assert get_s3_key_from_url("ftp://host/path/file.txt") == ""

    def test_empty_string_returns_empty(self):
        assert get_s3_key_from_url("") == ""

    def test_https_leading_slash_stripped(self):
        url = "https://bucket.s3.amazonaws.com/key.json"
        assert not get_s3_key_from_url(url).startswith("/")

    def test_s3_scheme_with_encoded_chars(self):
        url = "s3://my-bucket/some%20folder/file%2Bname.pdf"
        assert "some folder" in get_s3_key_from_url(url)


# ===========================================================================
# Section 2 – S3Utility.__init__
# ===========================================================================
class TestS3UtilityInit:
    def test_bucket_name_read_from_env(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        with patch("src.utils.s3_utility.boto3") as mock_boto:
            mock_boto.client.return_value = MagicMock()
            util = S3Utility()
        assert util.bucket_name == "my-bucket"

    def test_boto3_client_created_with_region(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
        monkeypatch.setenv("S3_BUCKET_NAME", "b")
        with patch("src.utils.s3_utility.boto3") as mock_boto:
            mock_boto.client.return_value = MagicMock()
            S3Utility()
        mock_boto.client.assert_called_once_with("s3", region_name="ap-southeast-1")

    def test_lambda_region_takes_priority_over_aws_region(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION_LAMBDA", "us-west-2")
        monkeypatch.setenv("AWS_REGION",        "eu-central-1")
        monkeypatch.setenv("S3_BUCKET_NAME",    "b")
        with patch("src.utils.s3_utility.boto3") as mock_boto:
            mock_boto.client.return_value = MagicMock()
            S3Utility()
        _, kwargs = mock_boto.client.call_args
        assert kwargs["region_name"] == "us-west-2"


# ===========================================================================
# Section 3 – upload_file
# ===========================================================================
class TestUploadFile:
    def test_success_returns_s3_url(self, s3util):
        s3util.s3_client.put_object.return_value = {}
        result = s3util.upload_file(b"data", "report.pdf", "reports")
        assert result == "s3://test-bucket/reports/report.pdf"

    def test_calls_put_object_with_correct_args(self, s3util):
        s3util.s3_client.put_object.return_value = {}
        s3util.upload_file(b"hello", "test.txt", "docs")
        s3util.s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="docs/test.txt",
            Body=b"hello",
        )

    def test_client_error_raises_500(self, s3util):
        s3util.s3_client.put_object.side_effect = make_client_error("AccessDenied")
        with pytest.raises(HTTPException) as exc_info:
            s3util.upload_file(b"data", "f.txt", "folder")
        assert exc_info.value.status_code == 500
        assert "Failed to upload file to S3" in exc_info.value.detail


# ===========================================================================
# Section 4 – upload_file_by_url
# ===========================================================================
class TestUploadFileByUrl:
    def test_success_returns_original_url(self, s3util):
        s3util.s3_client.put_object.return_value = {}
        url = "https://test-bucket.s3.amazonaws.com/folder/file.txt"
        assert s3util.upload_file_by_url(b"content", url) == url

    def test_calls_put_object_with_extracted_key(self, s3util):
        s3util.s3_client.put_object.return_value = {}
        url = "https://test-bucket.s3.amazonaws.com/docs/report.pdf"
        s3util.upload_file_by_url(b"pdf", url)
        s3util.s3_client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="docs/report.pdf",
            Body=b"pdf",
        )

    def test_client_error_raises_500(self, s3util):
        s3util.s3_client.put_object.side_effect = make_client_error("InternalError")
        with pytest.raises(HTTPException) as exc_info:
            s3util.upload_file_by_url(b"data", "https://b.s3.amazonaws.com/k")
        assert exc_info.value.status_code == 500


# ===========================================================================
# Section 5 – get_data_from_s3_by_url
# ===========================================================================
class TestGetDataFromS3ByUrl:
    def _mock_body(self, data: bytes) -> MagicMock:
        body = MagicMock()
        body.read.return_value = data
        return body

    def test_returns_bytes_on_success(self, s3util):
        s3util.s3_client.get_object.return_value = {
            "Body": self._mock_body(b"file-content")
        }
        result = s3util.get_data_from_s3_by_url(
            "https://test-bucket.s3.amazonaws.com/x/y.txt"
        )
        assert result == b"file-content"

    def test_no_such_key_raises_404(self, s3util):
        s3util.s3_client.get_object.side_effect = make_client_error("NoSuchKey")
        with pytest.raises(HTTPException) as exc_info:
            s3util.get_data_from_s3_by_url(
                "https://test-bucket.s3.amazonaws.com/missing.txt"
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "File not found in S3"

    def test_access_denied_raises_404(self, s3util):
        s3util.s3_client.get_object.side_effect = make_client_error("AccessDenied")
        with pytest.raises(HTTPException) as exc_info:
            s3util.get_data_from_s3_by_url(
                "https://test-bucket.s3.amazonaws.com/private.txt"
            )
        assert exc_info.value.status_code == 404

    def test_other_client_error_raises_500(self, s3util):
        s3util.s3_client.get_object.side_effect = make_client_error(
            "ServiceUnavailable"
        )
        with pytest.raises(HTTPException) as exc_info:
            s3util.get_data_from_s3_by_url(
                "https://test-bucket.s3.amazonaws.com/k"
            )
        assert exc_info.value.status_code == 500

    def test_url_decoded_before_key_extraction(self, s3util):
        s3util.s3_client.get_object.return_value = {
            "Body": self._mock_body(b"data")
        }
        s3util.get_data_from_s3_by_url(
            "https://test-bucket.s3.amazonaws.com/path%2Ffile.txt"
        )
        called_key = s3util.s3_client.get_object.call_args[1]["Key"]
        assert "path" in called_key


# ===========================================================================
# Section 6 – get_file
# ===========================================================================
class TestGetFile:
    def _mock_body(self, data: bytes) -> MagicMock:
        body = MagicMock()
        body.read.return_value = data
        return body

    def test_returns_bytes_on_success(self, s3util):
        s3util.s3_client.get_object.return_value = {
            "Body": self._mock_body(b"data")
        }
        assert s3util.get_file("report.pdf", "docs") == b"data"

    def test_constructs_correct_s3_key(self, s3util):
        s3util.s3_client.get_object.return_value = {
            "Body": self._mock_body(b"")
        }
        s3util.get_file("file.txt", "folder")
        s3util.s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="folder/file.txt"
        )

    def test_no_such_key_raises_404(self, s3util):
        s3util.s3_client.get_object.side_effect = make_client_error("NoSuchKey")
        with pytest.raises(HTTPException) as exc_info:
            s3util.get_file("missing.pdf", "docs")
        assert exc_info.value.status_code == 404

    def test_other_client_error_raises_500(self, s3util):
        s3util.s3_client.get_object.side_effect = make_client_error("InternalError")
        with pytest.raises(HTTPException) as exc_info:
            s3util.get_file("file.pdf", "docs")
        assert exc_info.value.status_code == 500


# ===========================================================================
# Section 7 – _get_s3_object
# ===========================================================================
class TestGetS3Object:
    def test_returns_bytes_on_success(self, s3util):
        body = MagicMock()
        body.read.return_value = b"raw"
        s3util.s3_client.get_object.return_value = {"Body": body}
        assert s3util._get_s3_object("some/key.txt") == b"raw"

    def test_client_error_raises_500_with_message(self, s3util):
        s3util.logger = MagicMock()
        s3util.s3_client.get_object.side_effect = make_client_error("InternalError")
        with pytest.raises(HTTPException) as exc_info:
            s3util._get_s3_object("bad/key.txt")
        assert exc_info.value.status_code == 500
        assert "Failed to retrieve S3 object" in exc_info.value.detail


# ===========================================================================
# Section 8 – generate_presigned_url
# ===========================================================================
class TestGeneratePresignedUrl:
    def test_returns_presigned_url(self, s3util):
        s3util.s3_client.generate_presigned_url.return_value = (
            "https://presigned.url/token"
        )
        result = s3util.generate_presigned_url("s3://test-bucket/folder/file.pdf")
        assert result == "https://presigned.url/token"

    def test_calls_with_correct_params(self, s3util):
        s3util.s3_client.generate_presigned_url.return_value = "https://url"
        s3util.generate_presigned_url(
            "s3://test-bucket/docs/report.pdf", expiration=3600
        )
        s3util.s3_client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "docs/report.pdf"},
            ExpiresIn=3600,
        )

    def test_default_expiration_is_7_days(self, s3util):
        s3util.s3_client.generate_presigned_url.return_value = "https://url"
        s3util.generate_presigned_url("s3://test-bucket/key.txt")
        _, kwargs = s3util.s3_client.generate_presigned_url.call_args
        assert kwargs["ExpiresIn"] == 604800

    def test_no_such_key_raises_404(self, s3util):
        s3util.s3_client.generate_presigned_url.side_effect = make_client_error(
            "NoSuchKey"
        )
        with pytest.raises(HTTPException) as exc_info:
            s3util.generate_presigned_url("s3://test-bucket/missing.pdf")
        assert exc_info.value.status_code == 404

    def test_other_client_error_raises_500(self, s3util):
        s3util.s3_client.generate_presigned_url.side_effect = make_client_error(
            "InternalError"
        )
        with pytest.raises(HTTPException) as exc_info:
            s3util.generate_presigned_url("s3://test-bucket/key.txt")
        assert exc_info.value.status_code == 500


# ===========================================================================
# Section 9 – delete_file
# ===========================================================================
class TestDeleteFile:
    def test_returns_success_dict(self, s3util):
        s3util.s3_client.delete_object.return_value = {}
        result = s3util.delete_file("report.pdf", "docs")
        assert result["status"] == "success"
        assert "report.pdf" in result["message"]

    def test_calls_delete_object_with_correct_key(self, s3util):
        s3util.s3_client.delete_object.return_value = {}
        s3util.delete_file("file.txt", "folder")
        s3util.s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="folder/file.txt"
        )

    def test_no_such_key_raises_404(self, s3util):
        s3util.s3_client.delete_object.side_effect = make_client_error("NoSuchKey")
        with pytest.raises(HTTPException) as exc_info:
            s3util.delete_file("missing.pdf", "docs")
        assert exc_info.value.status_code == 404

    def test_other_client_error_raises_500(self, s3util):
        s3util.s3_client.delete_object.side_effect = make_client_error("AccessDenied")
        with pytest.raises(HTTPException) as exc_info:
            s3util.delete_file("file.pdf", "docs")
        assert exc_info.value.status_code == 500


# ===========================================================================
# Section 10 – delete_file_by_url
# ===========================================================================
class TestDeleteFileByUrl:
    def test_returns_success_dict(self, s3util):
        s3util.s3_client.delete_object.return_value = {}
        result = s3util.delete_file_by_url("s3://test-bucket/folder/file.txt")
        assert result["status"] == "success"

    def test_extracts_key_and_calls_delete_object(self, s3util):
        s3util.s3_client.delete_object.return_value = {}
        s3util.delete_file_by_url("s3://test-bucket/folder/file.txt")
        s3util.s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="folder/file.txt"
        )

    def test_no_such_key_raises_404(self, s3util):
        s3util.s3_client.delete_object.side_effect = make_client_error("NoSuchKey")
        with pytest.raises(HTTPException) as exc_info:
            s3util.delete_file_by_url("s3://test-bucket/missing.txt")
        assert exc_info.value.status_code == 404

    def test_other_client_error_raises_500(self, s3util):
        s3util.s3_client.delete_object.side_effect = make_client_error("InternalError")
        with pytest.raises(HTTPException) as exc_info:
            s3util.delete_file_by_url("s3://test-bucket/file.txt")
        assert exc_info.value.status_code == 500


# ===========================================================================
# Section 11 – create_zip_and_upload_for_urls
# ===========================================================================
class TestCreateZipAndUploadForUrls:
    def test_creates_zip_and_uploads(self, s3util):
        """Every file URL is fetched, added to a zip, and the zip is uploaded."""
        def fake_get(url):
            return b"content_a" if "a.txt" in url else b"content_b"

        s3util.get_data_from_s3_by_url = MagicMock(side_effect=fake_get)
        s3util.s3_client.put_object.return_value = {}

        result = s3util.create_zip_and_upload_for_urls(
            [
                "s3://test-bucket/folder/a.txt",
                "s3://test-bucket/folder/b.txt",
            ],
            "zips",
            "output.zip",
        )

        assert result == "s3://test-bucket/zips/output.zip"
        assert s3util.get_data_from_s3_by_url.call_count == 2

    def test_zip_contains_correct_filenames(self, s3util):
        """Files inside the zip are named after the last URL segment."""
        s3util.get_data_from_s3_by_url = MagicMock(return_value=b"data")
        captured = {}

        def capture_upload(content, filename, folder):
            captured["zip_bytes"] = content
            return f"s3://test-bucket/{folder}/{filename}"

        s3util.upload_file = capture_upload

        s3util.create_zip_and_upload_for_urls(
            ["s3://test-bucket/docs/report.pdf"], "zips", "bundle.zip"
        )

        zf = zipfile.ZipFile(io.BytesIO(captured["zip_bytes"]))
        assert "report.pdf" in zf.namelist()


# ===========================================================================
# Section 12 – extract_filename_from_s3_url
# ===========================================================================
class TestExtractFilenameFromS3Url:
    def test_removes_uuid_suffix(self, s3util):
        url = "s3://bucket/folder/report_123e4567-e89b-12d3-a456-426614174000.pdf"
        assert s3util.extract_filename_from_s3_url(url) == "report.pdf"

    def test_plain_filename_unchanged(self, s3util):
        assert (
            s3util.extract_filename_from_s3_url(
                "s3://bucket/folder/document.docx"
            )
            == "document.docx"
        )

    def test_https_url_extracts_filename(self, s3util):
        assert (
            s3util.extract_filename_from_s3_url(
                "https://bucket.s3.amazonaws.com/folder/data.csv"
            )
            == "data.csv"
        )

    def test_url_encoded_characters_decoded(self, s3util):
        result = s3util.extract_filename_from_s3_url(
            "s3://bucket/folder/my%20file.txt"
        )
        assert "my file" in result

    def test_filename_with_underscores_but_no_uuid_unchanged(self, s3util):
        assert (
            s3util.extract_filename_from_s3_url(
                "s3://bucket/folder/my_report_final.pdf"
            )
            == "my_report_final.pdf"
        )


# ===========================================================================
# Section 13 – copy_s3_file_to_new_path
# ===========================================================================
class TestCopyS3FileToNewPath:
    def _setup_copy(self, s3util):
        s3util.s3_client.copy_object.return_value = {}
        s3util.s3_client.generate_presigned_url.return_value = (
            "https://presigned/url"
        )

    def test_copies_virtual_hosted_https_url(self, s3util):
        self._setup_copy(s3util)
        presigned, filename = s3util.copy_s3_file_to_new_path(
            "https://test-bucket.s3.amazonaws.com/src/file.pdf", "dest-folder"
        )
        assert presigned == "https://presigned/url"
        assert filename == "file.pdf"

    def test_copies_s3_scheme_url(self, s3util):
        self._setup_copy(s3util)
        presigned, filename = s3util.copy_s3_file_to_new_path(
            "s3://some-bucket/src/archive.zip", "backups"
        )
        assert presigned == "https://presigned/url"
        assert filename == "archive.zip"

    def test_spaces_in_filename_replaced_with_underscores(self, s3util):
        self._setup_copy(s3util)
        _, filename = s3util.copy_s3_file_to_new_path(
            "s3://bucket/src/my file.txt", "dst"
        )
        assert " " not in filename

    def test_exception_raises_value_error(self, s3util):
        s3util.s3_client.copy_object.side_effect = Exception("network error")
        with pytest.raises(ValueError, match="S3 copy operation failed"):
            s3util.copy_s3_file_to_new_path("s3://bucket/key.txt", "dst")

    def test_copy_object_called_with_correct_source(self, s3util):
        self._setup_copy(s3util)
        s3util.copy_s3_file_to_new_path(
            "s3://src-bucket/path/file.txt", "new-folder"
        )
        copy_call = s3util.s3_client.copy_object.call_args
        assert copy_call[1]["CopySource"] == {
            "Bucket": "src-bucket",
            "Key": "path/file.txt",
        }
        assert copy_call[1]["Bucket"] == "test-bucket"
        assert copy_call[1]["Key"] == "new-folder/file.txt"


# ===========================================================================
# Section 14 – _parse_virtual_hosted_style
# ===========================================================================
class TestParseVirtualHostedStyle:
    def test_returns_bucket_and_key(self, s3util):
        parsed = urlparse(
            "https://my-bucket.s3.amazonaws.com/folder/key.txt"
        )
        bucket, key = s3util._parse_virtual_hosted_style(parsed)
        assert key == "folder/key.txt"
        assert bucket == "test-bucket"

    def test_empty_key_raises_value_error(self, s3util):
        parsed = urlparse("https://my-bucket.s3.amazonaws.com/")
        with pytest.raises(ValueError, match="No file key found in URL"):
            s3util._parse_virtual_hosted_style(parsed)


# ===========================================================================
# Section 15 – _parse_path_style
# ===========================================================================
class TestParsePathStyle:
    def test_returns_bucket_and_key(self, s3util):
        parsed = urlparse(
            "https://s3.amazonaws.com/my-bucket/folder/file.txt"
        )
        bucket, key = s3util._parse_path_style(parsed)
        assert bucket == "my-bucket"
        assert key == "folder/file.txt"

    def test_empty_key_raises_value_error(self, s3util):
        parsed = urlparse("https://s3.amazonaws.com/my-bucket")
        with pytest.raises(ValueError, match="No file key found in URL"):
            s3util._parse_path_style(parsed)


# ===========================================================================
# Section 16 – _parse_s3_url_for_copy
# ===========================================================================
class TestParseS3UrlForCopy:
    def test_https_virtual_hosted_style(self, s3util):
        bucket, key = s3util._parse_s3_url_for_copy(
            "https://my-bucket.s3.amazonaws.com/folder/file.pdf"
        )
        assert key == "folder/file.pdf"

    def test_https_path_style(self, s3util):
        bucket, key = s3util._parse_s3_url_for_copy(
            "https://s3.amazonaws.com/my-bucket/folder/file.pdf"
        )
        assert bucket == "my-bucket"
        assert key == "folder/file.pdf"

    def test_s3_scheme_success(self, s3util):
        bucket, key = s3util._parse_s3_url_for_copy(
            "s3://my-bucket/folder/file.txt"
        )
        assert bucket == "my-bucket"
        assert key == "folder/file.txt"

    def test_s3_scheme_no_key_raises(self, s3util):
        with pytest.raises(ValueError, match="No file key found in URL"):
            s3util._parse_s3_url_for_copy("s3://my-bucket")

    def test_https_non_amazonaws_domain_raises(self, s3util):
        with pytest.raises(ValueError, match="Invalid S3 URL format"):
            s3util._parse_s3_url_for_copy(
                "https://example.com/bucket/key.txt"
            )

    def test_https_amazonaws_but_unrecognised_subdomain_raises(self, s3util):
        with pytest.raises(ValueError, match="Invalid S3 URL format"):
            s3util._parse_s3_url_for_copy(
                "https://unknown.amazonaws.com/bucket/key.txt"
            )

    def test_unsupported_scheme_raises(self, s3util):
        with pytest.raises(ValueError, match="Unsupported URL format"):
            s3util._parse_s3_url_for_copy("ftp://my-bucket/folder/key.txt")

    def test_https_s3_dash_region_path_style(self, s3util):
        bucket, key = s3util._parse_s3_url_for_copy(
            "https://s3-us-east-1.amazonaws.com/my-bucket/folder/file.txt"
        )
        assert key == "folder/file.txt"