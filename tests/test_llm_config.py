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


def test_provider_budget_rejection_is_fail_fast():
    class Error402(Exception):
        status_code = 402

    class Completions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise Error402("Error code: 402 - Insufficient Balance")

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": Completions()})()

    llm = object.__new__(llm_module.OpenAILLM)
    llm.model = "deepseek-flash"
    llm.max_retries = 3
    llm.last_usage = {}
    llm._client = Client()
    try:
        llm.chat([{"role": "user", "content": "ping"}])
    except llm_module.ProviderBudgetError as exc:
        assert "insufficient" in str(exc).lower()
    else:
        raise AssertionError("expected non-retryable ProviderBudgetError")
    assert llm._client.chat.completions.calls == 1
