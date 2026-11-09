"""Streaming ``multipart/form-data`` parser for the local web app.

This replaces ``cgi.FieldStorage``: the ``cgi`` module was removed from the
standard library in Python 3.13. File parts are streamed straight to disk in
1 MB chunks, so a large CCTV upload never has to fit in memory.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from email.message import Message
from email.utils import collapse_rfc2231_value
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024
MAX_FIELD_BYTES = 1024 * 1024


class MultipartError(ValueError):
    """Raised when a request body is not valid multipart/form-data."""


@dataclass
class UploadedFile:
    field_name: str
    filename: str
    path: Path
    size: int


@dataclass
class MultipartForm:
    fields: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, list[UploadedFile]] = field(default_factory=dict)

    def first_file(self, name: str) -> UploadedFile | None:
        items = self.files.get(name)
        return items[0] if items else None

    def cleanup(self) -> None:
        """Delete temporary files that were not moved somewhere else."""
        for items in self.files.values():
            for item in items:
                item.path.unlink(missing_ok=True)


def get_boundary(content_type: str) -> bytes:
    message = Message()
    message["content-type"] = content_type
    if message.get_content_type() != "multipart/form-data":
        raise MultipartError("Expected a multipart/form-data request.")
    boundary = message.get_param("boundary")
    if not boundary or not isinstance(boundary, str):
        raise MultipartError("The multipart request has no boundary.")
    if len(boundary) > 200:
        raise MultipartError("The multipart boundary is too long.")
    return boundary.encode("ascii")


def _parse_part_headers(raw: bytes) -> tuple[str, str | None]:
    message = Message()
    for line in raw.decode("utf-8", errors="replace").split("\r\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            message[key.strip()] = value.strip()

    if not message.get("content-disposition", "").lower().startswith("form-data"):
        raise MultipartError("A multipart section is missing 'Content-Disposition: form-data'.")
    name = message.get_param("name", header="content-disposition")
    filename = message.get_param("filename", header="content-disposition")
    if not name:
        raise MultipartError("A multipart section has no field name.")
    name = collapse_rfc2231_value(name)
    filename = collapse_rfc2231_value(filename) if filename is not None else None
    return name, filename


class _BodyReader:
    def __init__(self, stream: BinaryIO, length: int) -> None:
        self.stream = stream
        self.remaining = length
        self.buffer = b""

    def fill(self) -> bool:
        if self.remaining <= 0:
            return False
        chunk = self.stream.read(min(CHUNK_SIZE, self.remaining))
        if not chunk:
            self.remaining = 0
            return False
        self.remaining -= len(chunk)
        self.buffer += chunk
        return True

    def require(self, size: int) -> None:
        while len(self.buffer) < size:
            if not self.fill():
                raise MultipartError("The upload ended unexpectedly.")


def parse_multipart(
    stream: BinaryIO,
    content_type: str,
    content_length: int,
    tmp_dir: Path,
) -> MultipartForm:
    """Parse a multipart body of exactly ``content_length`` bytes from ``stream``."""
    if content_length <= 0:
        raise MultipartError("The upload request is empty.")

    delimiter = b"--" + get_boundary(content_type)
    part_end = b"\r\n" + delimiter
    reader = _BodyReader(stream, content_length)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    form = MultipartForm()

    try:
        # Skip the preamble up to the first delimiter.
        while (index := reader.buffer.find(delimiter)) < 0:
            reader.buffer = reader.buffer[-len(delimiter) :]
            if not reader.fill():
                raise MultipartError("The upload does not contain the multipart boundary.")
        reader.buffer = reader.buffer[index + len(delimiter) :]

        while True:
            reader.require(2)
            if reader.buffer.startswith(b"--"):
                break  # closing delimiter
            while (line_end := reader.buffer.find(b"\r\n")) < 0:
                reader.require(len(reader.buffer) + 1)
            reader.buffer = reader.buffer[line_end + 2 :]

            while (header_end := reader.buffer.find(b"\r\n\r\n")) < 0:
                if len(reader.buffer) > MAX_HEADER_BYTES:
                    raise MultipartError("A multipart section header is too large.")
                reader.require(len(reader.buffer) + 1)
            name, filename = _parse_part_headers(reader.buffer[:header_end])
            reader.buffer = reader.buffer[header_end + 4 :]

            if filename is not None:
                _read_file_part(reader, part_end, form, name, filename, tmp_dir)
            else:
                _read_field_part(reader, part_end, form, name)
    except BaseException:
        form.cleanup()
        raise
    return form


def _stream_until(reader: _BodyReader, part_end: bytes, write) -> None:
    while True:
        index = reader.buffer.find(part_end)
        if index >= 0:
            write(reader.buffer[:index])
            reader.buffer = reader.buffer[index + len(part_end) :]
            return
        # Keep enough bytes to recognise a delimiter split across two reads.
        safe = len(reader.buffer) - len(part_end) + 1
        if safe > 0:
            write(reader.buffer[:safe])
            reader.buffer = reader.buffer[safe:]
        if not reader.fill():
            raise MultipartError("The upload ended before the closing boundary.")


def _read_file_part(
    reader: _BodyReader,
    part_end: bytes,
    form: MultipartForm,
    name: str,
    filename: str,
    tmp_dir: Path,
) -> None:
    handle = tempfile.NamedTemporaryFile(dir=tmp_dir, prefix="upload_", suffix=".part", delete=False)
    upload = UploadedFile(field_name=name, filename=filename, path=Path(handle.name), size=0)
    form.files.setdefault(name, []).append(upload)
    with handle:

        def write(data: bytes) -> None:
            handle.write(data)
            upload.size += len(data)

        _stream_until(reader, part_end, write)


def _read_field_part(reader: _BodyReader, part_end: bytes, form: MultipartForm, name: str) -> None:
    value = bytearray()

    def write(data: bytes) -> None:
        value.extend(data)
        if len(value) > MAX_FIELD_BYTES:
            raise MultipartError(f"Form field '{name}' is too large.")

    _stream_until(reader, part_end, write)
    form.fields.setdefault(name, []).append(value.decode("utf-8", errors="replace"))
