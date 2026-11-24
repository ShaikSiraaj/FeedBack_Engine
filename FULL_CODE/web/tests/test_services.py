import pytest

from services import translation_service


class FakeLanguage:
    def __init__(self, code, name):
        self.code = code
        self.name = name
        self.translations = {}

    def get_translation(self, target):
        return self.translations.get(target.code)


class FakeTranslation:
    def translate(self, text):
        return f"translated: {text}"


class EmptyTranslation:
    def translate(self, text):
        return ""


def test_translation_uses_installed_local_pair(monkeypatch):
    english = FakeLanguage("en", "English")
    spanish = FakeLanguage("es", "Spanish")
    english.translations["es"] = FakeTranslation()
    monkeypatch.setattr(
        translation_service,
        "_installed_languages",
        lambda: {"en": english, "es": spanish},
    )

    assert translation_service.translate_text("hello", "en", "es") == "translated: hello"


def test_translation_languages_reports_only_installed_pairs(monkeypatch):
    english = FakeLanguage("en", "English")
    spanish = FakeLanguage("es", "Spanish")
    english.translations["es"] = FakeTranslation()
    monkeypatch.setattr(
        translation_service,
        "_installed_languages",
        lambda: {"en": english, "es": spanish},
    )

    result = translation_service.get_translation_languages()

    assert result["pairs"] == [{"source": "en", "target": "es"}]
    assert {language["code"] for language in result["languages"]} == {"en", "es"}


def test_translation_rejects_empty_text():
    with pytest.raises(ValueError, match="required"):
        translation_service.translate_text(" ", "en", "es")


def test_translation_rejects_romanized_hindi(monkeypatch):
    hindi = FakeLanguage("hi", "Hindi")
    english = FakeLanguage("en", "English")
    monkeypatch.setattr(
        translation_service,
        "_installed_languages",
        lambda: {"hi": hindi, "en": english},
    )

    with pytest.raises(ValueError, match="expects Devanagari"):
        translation_service.translate_text(
            "saffron customer care mein aapka swagat hai", "hi", "en"
        )


@pytest.mark.parametrize(
    "emoji",
    [
        "\U0001f44d\U0001f60d",
        "\U0001f468\u200d\U0001f469\u200d\U0001f467",
        "\U0001f1ee\U0001f1f3",
        "1\ufe0f\u20e3",
    ],
)
def test_translation_preserves_emoji_sequences(monkeypatch, emoji):
    english = FakeLanguage("en", "English")
    spanish = FakeLanguage("es", "Spanish")
    english.translations["es"] = FakeTranslation()
    monkeypatch.setattr(
        translation_service,
        "_installed_languages",
        lambda: {"en": english, "es": spanish},
    )

    result = translation_service.translate_text(
        f"Really nice {emoji} Keep it up!", "en", "es"
    )

    assert emoji in result
    assert result.count(emoji) == 1
    assert result.startswith("translated: Really nice ")
    assert result.endswith("translated: Keep it up!")


def test_translation_reports_when_no_local_languages_are_installed(monkeypatch):
    monkeypatch.setattr(translation_service, "_installed_languages", lambda: {})

    with pytest.raises(translation_service.TranslationUnavailableError, match="No local"):
        translation_service.get_translation_languages()


def test_translation_reports_missing_pair(monkeypatch):
    english = FakeLanguage("en", "English")
    spanish = FakeLanguage("es", "Spanish")
    monkeypatch.setattr(
        translation_service,
        "_installed_languages",
        lambda: {"en": english, "es": spanish},
    )

    with pytest.raises(translation_service.TranslationUnavailableError, match="No local"):
        translation_service.translate_text("hello", "en", "es")


def test_translation_rejects_empty_model_output(monkeypatch):
    english = FakeLanguage("en", "English")
    spanish = FakeLanguage("es", "Spanish")
    english.translations["es"] = EmptyTranslation()
    monkeypatch.setattr(
        translation_service,
        "_installed_languages",
        lambda: {"en": english, "es": spanish},
    )

    with pytest.raises(translation_service.TranslationUnavailableError, match="empty result"):
        translation_service.translate_text("hello", "en", "es")
