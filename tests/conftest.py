"""The normal release suite must never contact a real provider."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_external_connections(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("External connections are forbidden in the offline test suite")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
