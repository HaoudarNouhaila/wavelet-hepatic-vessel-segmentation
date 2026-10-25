import io
import tarfile

from src.data.remote_tar import build_index, download_member


class BytesReader:
    def __init__(self, data: bytes):
        self.data = data
        self.calls = 0

    def read(self, start: int, length: int) -> bytes:
        self.calls += 1
        return self.data[start : start + length]


def make_tar(files: dict[str, bytes], fmt=tarfile.PAX_FORMAT) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=fmt) as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_index_offsets_point_to_file_contents():
    files = {
        "Task08_HepaticVessel/dataset.json": b'{"a": 1}',
        "Task08_HepaticVessel/imagesTr/hepaticvessel_001.nii.gz": bytes(range(256)) * 9,
        "Task08_HepaticVessel/labelsTr/._hepaticvessel_001.nii.gz": b"x",
        "Task08_HepaticVessel/" + "very_long_directory_name/" * 6 + "file.nii.gz": b"long name",
    }
    for fmt in (tarfile.PAX_FORMAT, tarfile.GNU_FORMAT):
        data = make_tar(files, fmt)
        members = build_index(BytesReader(data), chunk=1024)
        found = {m.name: data[m.offset : m.offset + m.size] for m in members}
        assert found == files


def test_download_member(tmp_path, monkeypatch):
    files = {"a/b.bin": bytes(range(200)) * 50}
    data = make_tar(files)
    member = build_index(BytesReader(data))[0]
    reader = BytesReader(data)
    monkeypatch.setattr("src.data.remote_tar.RangeReader", lambda url: reader)
    out = tmp_path / "b.bin"
    download_member("http://unused", member, out, chunk=777)
    assert out.read_bytes() == files["a/b.bin"]
    calls = reader.calls
    download_member("http://unused", member, out)  # déjà présent : aucune requête
    assert reader.calls == calls
