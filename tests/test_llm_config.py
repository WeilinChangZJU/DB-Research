import bemula.llm as llm_module


def test_effective_environment_is_loaded_once_without_override(monkeypatch):
    calls = []

    def fake_load_dotenv(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(llm_module, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(llm_module, "_ENV_LOADED", False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://frozen.example")
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")

    assert llm_module.resolve_endpoint("deepseek-flash") == (
        "https://frozen.example",
        "process-key",
    )
    # A later endpoint resolution must not re-read a changed .env file.
    assert llm_module.resolve_endpoint("deepseek-flash") == (
        "https://frozen.example",
        "process-key",
    )
    assert calls == [{"override": False}]
