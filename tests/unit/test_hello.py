import pytest

from hello import greet


@pytest.mark.unit
class TestGreet:
    def test_default(self):
        assert greet() == "Hello, world!"

    def test_custom_name(self):
        assert greet("Alice") == "Hello, Alice!"
