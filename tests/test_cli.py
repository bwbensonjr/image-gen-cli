from __future__ import annotations

from pathlib import Path

import pytest

from image_gen_cli import cli, generate


def _set_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "image-gen" in capsys.readouterr().out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "image-gen 0.1.0" in capsys.readouterr().out


def test_missing_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    rc = cli.main(["a cat"])

    assert rc == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def test_main_happy_path_prints_paths_to_stdout(monkeypatch, capsys, tmp_path):
    _set_api_key(monkeypatch)
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run(prompt, files, base, client=None, verbose=False, image_model="gpt-image-2"):
        calls["verbose"] = verbose
        calls["image_model"] = image_model
        return [Path("out-1.png")]

    monkeypatch.setattr(generate, "run", fake_run)

    rc = cli.main(["-o", "out", "a cat"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "out-1.png"
    assert "Generating image..." in captured.err
    assert calls["verbose"] is False
    assert calls["image_model"] == "gpt-image-2"


def test_main_image_model_override(monkeypatch, capsys, tmp_path):
    _set_api_key(monkeypatch)
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run(prompt, files, base, client=None, verbose=False, image_model="gpt-image-2"):
        calls["image_model"] = image_model
        return [Path("out-1.png")]

    monkeypatch.setattr(generate, "run", fake_run)

    rc = cli.main(["--image-model", "gpt-image-1.5", "a cat"])

    assert rc == 0
    assert calls["image_model"] == "gpt-image-1.5"


def test_main_verbose_passes_verbose_to_run(monkeypatch, capsys, tmp_path):
    _set_api_key(monkeypatch)
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run(prompt, files, base, client=None, verbose=False, image_model="gpt-image-2"):
        calls["verbose"] = verbose
        calls["image_model"] = image_model
        return [Path("out-1.png")]

    monkeypatch.setattr(generate, "run", fake_run)

    rc = cli.main(["-v", "a cat"])

    assert rc == 0
    assert calls["verbose"] is True


def test_main_missing_input_file_exits_2(monkeypatch, capsys, tmp_path):
    _set_api_key(monkeypatch)
    monkeypatch.chdir(tmp_path)

    rc = cli.main(["-i", "missing.md", "a cat"])

    assert rc == 2
    assert "input file not found" in capsys.readouterr().err


def test_main_verbose_prints_traceback(monkeypatch, capsys):
    _set_api_key(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("kapow")

    monkeypatch.setattr(generate, "run", boom)

    rc = cli.main(["-v", "a cat"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" in err
    assert "kapow" in err


def test_main_non_verbose_hides_traceback(monkeypatch, capsys):
    _set_api_key(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("kapow")

    monkeypatch.setattr(generate, "run", boom)

    rc = cli.main(["a cat"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" not in err
    assert "re-run with --verbose" in err


def test_main_maps_authentication_error_to_exit_1(monkeypatch, capsys):
    _set_api_key(monkeypatch)
    from openai import AuthenticationError

    def boom(*a, **kw):
        raise AuthenticationError(
            message="bad key",
            response=_make_dummy_response(401),
            body=None,
        )

    monkeypatch.setattr(generate, "run", boom)

    rc = cli.main(["a cat"])

    assert rc == 1
    assert "OpenAI rejected the API key" in capsys.readouterr().err


def test_main_no_image_returned_exits_1(monkeypatch, capsys):
    _set_api_key(monkeypatch)

    def boom(*a, **kw):
        raise generate.NoImageReturnedError("model returned no image. Try rephrasing the prompt.")

    monkeypatch.setattr(generate, "run", boom)

    rc = cli.main(["a cat"])

    assert rc == 1
    assert "model returned no image" in capsys.readouterr().err


def _make_dummy_response(status: int):
    import httpx

    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
