def main() -> None:
    import sys
    sys.path.insert(0, "backend")
    from app.main import app  # noqa: F401
    print("app title:", app.title)
    print("version:", app.version)
    print("routes:", sorted({r.path for r in app.routes if r.path.startswith("/api")}))
    print("IMPORT_OK")


if __name__ == "__main__":
    main()
