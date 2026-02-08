def greet(name: str = "world") -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":  # pragma: no cover
    print(greet())
