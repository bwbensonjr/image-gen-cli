from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def make_response(image_b64_list: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        output=[
            SimpleNamespace(type="image_generation_call", result=b64)
            for b64 in image_b64_list
        ]
    )


@pytest.fixture
def fake_openai_client():
    client = MagicMock()
    client.responses.create.return_value = make_response(["aGVsbG8="])
    return client


@pytest.fixture
def make_response_fixture():
    return make_response
