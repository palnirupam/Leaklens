def main() -> None:
    import sys
    sys.path.insert(0, "backend")
    from app.core.config import settings
    print("data_dir:", settings.data_dir)
    print("db_path:", settings.db_path)
    print("uploads_dir:", settings.uploads_dir)
    print("reports_dir:", settings.reports_dir)


if __name__ == "__main__":
    main()
