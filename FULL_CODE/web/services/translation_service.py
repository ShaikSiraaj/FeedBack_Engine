"""Offline neural translation backed by locally installed Argos language packages."""

from functools import lru_cache

import regex

_EMOJI_PATTERN = regex.compile(
    r"(?:[#*0-9]\uFE0F?\u20E3)"
    r"|(?:\p{Extended_Pictographic}\uFE0F?(?:\p{Emoji_Modifier})?"
    r"(?:\u200D\p{Extended_Pictographic}\uFE0F?(?:\p{Emoji_Modifier})?)*)"
    r"|(?:\p{Regional_Indicator}{2})"
)


def _translate_segment(text, translation):
    content = text.strip()
    if not content:
        return text
    leading_whitespace = text[: len(text) - len(text.lstrip())]
    trailing_whitespace = text[len(text.rstrip()) :]
    return leading_whitespace + translation.translate(content) + trailing_whitespace


class TranslationUnavailableError(RuntimeError):
    """Raised when the offline translation runtime or requested model is unavailable."""


@lru_cache(maxsize=1)
def _installed_languages():
    try:
        import argostranslate.settings

        argostranslate.settings.chunk_type = argostranslate.settings.ChunkType.MINISBD
        import argostranslate.translate
    except ImportError as exc:
        raise TranslationUnavailableError(
            "Offline translation is unavailable. Install the web requirements first."
        ) from exc

    return {
        language.code: language
        for language in argostranslate.translate.get_installed_languages()
    }


def get_translation_languages():
    """Return local languages and translation pairs available on this machine."""
    languages = _installed_languages()
    if not languages:
        raise TranslationUnavailableError(
            "No local translation language packages are installed. Install an Argos "
            "language package, for example: argospm install translate-en_es."
        )

    pairs = []

    for source_code, source in languages.items():
        for target_code, target in languages.items():
            if source_code != target_code and source.get_translation(target):
                pairs.append({"source": source_code, "target": target_code})

    return {
        "languages": [
            {"code": language.code, "name": language.name}
            for language in sorted(languages.values(), key=lambda item: item.name)
        ],
        "pairs": pairs,
        "engine": "Argos Translate (local)",
    }


def translate_text(text, source_language, target_language):
    """Translate text using an already-installed local language package."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text to translate is required.")

    languages = _installed_languages()
    if not languages:
        raise TranslationUnavailableError(
            "No local translation language packages are installed."
        )

    source = languages.get(source_language)
    target = languages.get(target_language)
    if source is None or target is None:
        raise ValueError("Select languages available in the local language list.")

    if source_language == target_language:
        return text

    if source_language == "hi" and not any(
        "\u0900" <= character <= "\u097f" for character in text
    ):
        raise ValueError(
            "The installed Hindi model expects Devanagari text. Romanized Hindi "
            "or Hinglish transcripts are not supported."
        )

    translation = source.get_translation(target)
    if translation is None:
        raise TranslationUnavailableError(
            f"No local translation model is installed for "
            f"{source_language} to {target_language}."
        )

    translated_parts = []
    cursor = 0
    for match in _EMOJI_PATTERN.finditer(text):
        translated_parts.append(_translate_segment(text[cursor:match.start()], translation))
        translated_parts.append(match.group())
        cursor = match.end()
    translated_parts.append(_translate_segment(text[cursor:], translation))

    translated_text = "".join(translated_parts)
    if not translated_text or not translated_text.strip():
        raise TranslationUnavailableError(
            "The local translation model returned an empty result."
        )

    return translated_text
