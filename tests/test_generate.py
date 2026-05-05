from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_gen_cli import generate


def test_build_input_text_only_returns_string():
    assert generate.build_input("a cat", []) == "a cat"


def test_build_input_with_text_file_inlines_contents(tmp_path: Path):
    notes = tmp_path / "notes.md"
    notes.write_text("# Mood\nmoody")

    payload = generate.build_input("a cat", [notes])

    assert isinstance(payload, list)
    msg = payload[0]
    assert msg["role"] == "user"
    parts = msg["content"]
    assert parts[0] == {"type": "input_text", "text": "a cat"}
    assert parts[1]["type"] == "input_text"
    assert "<file: notes.md>" in parts[1]["text"]
    assert "moody" in parts[1]["text"]


def test_build_input_with_image_file_emits_data_url(tmp_path: Path):
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    payload = generate.build_input("a cat", [img])

    assert isinstance(payload, list)
    parts = payload[0]["content"]
    image_part = parts[1]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "auto"
    assert image_part["image_url"].startswith("data:image/png;base64,")


def test_build_input_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generate.build_input("x", [tmp_path / "missing.md"])


def test_write_images_numbers_from_one(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b64 = base64.b64encode(b"hello").decode()
    paths = generate.write_images([b64], "out")
    assert paths == [Path("out-1.png")]
    assert Path("out-1.png").read_bytes() == b"hello"


def test_write_images_overwrites_existing_file(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("out-1.png").write_bytes(b"stale")
    b64 = base64.b64encode(b"fresh").decode()
    generate.write_images([b64], "out")
    assert Path("out-1.png").read_bytes() == b"fresh"


def test_run_calls_client_with_expected_kwargs(tmp_path: Path, monkeypatch, fake_openai_client):
    monkeypatch.chdir(tmp_path)

    generate.run("a cat", [], "out", client=fake_openai_client)

    fake_openai_client.responses.create.assert_called_once()
    kwargs = fake_openai_client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["tools"] == [{"type": "image_generation", "model": "gpt-image-2"}]
    assert kwargs["input"] == "a cat"
    assert "reasoning" not in kwargs


def test_run_image_model_override(tmp_path: Path, monkeypatch, fake_openai_client):
    monkeypatch.chdir(tmp_path)

    generate.run(
        "a cat", [], "out", client=fake_openai_client, image_model="gpt-image-1.5"
    )

    kwargs = fake_openai_client.responses.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": "image_generation", "model": "gpt-image-1.5"}]


def test_run_verbose_requests_reasoning_and_dumps_output(
    tmp_path: Path, monkeypatch, fake_openai_client, capsys
):
    monkeypatch.chdir(tmp_path)
    from types import SimpleNamespace

    fake_openai_client.responses.create.return_value = SimpleNamespace(
        model="gpt-5.5",
        status="completed",
        usage=None,
        output=[
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                summary=[SimpleNamespace(text="thinking about otters", type="summary_text")],
                content=None,
            ),
            SimpleNamespace(
                type="image_generation_call",
                id="ig_1",
                status="completed",
                result="aGVsbG8=",
            ),
        ],
    )

    generate.run("a cat", [], "out", client=fake_openai_client, verbose=True)

    kwargs = fake_openai_client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"summary": "auto"}

    err = capsys.readouterr().err
    assert "model=gpt-5.5" in err
    assert "type=reasoning" in err
    assert "thinking about otters" in err
    assert "type=image_generation_call" in err


def test_run_raises_when_no_image_in_response(tmp_path: Path, monkeypatch, fake_openai_client):
    monkeypatch.chdir(tmp_path)
    fake_openai_client.responses.create.return_value = SimpleNamespace(output=[])

    with pytest.raises(generate.NoImageReturnedError):
        generate.run("a cat", [], "out", client=fake_openai_client)
