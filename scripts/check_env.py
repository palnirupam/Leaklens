def main() -> None:
    import fastapi, pydantic, pandas, numpy, sklearn, pydantic_settings
    try:
        import google.genai  # noqa: F401
        gemini = True
    except Exception:
        gemini = False
    print("fastapi", fastapi.__version__)
    print("pydantic", pydantic.__version__)
    print("pandas", pandas.__version__)
    print("numpy", numpy.__version__)
    print("sklearn", sklearn.__version__)
    print("pydantic_settings", pydantic_settings.__version__)
    print("google.genai present:", gemini)
    print("ALL_OK")


if __name__ == "__main__":
    main()
