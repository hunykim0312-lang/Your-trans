import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-6-astra"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 1
MAX_INPUT_CHARS = 5000
LANGUAGES = {
    "english": "🇺🇸 English",
    "japanese": "🇯🇵 日本語",
    "vietnamese": "🇻🇳 Tiếng Việt",
}
SYSTEM_PROMPT = """You are a professional translator.
Detect the source language automatically and translate the user's text into English, Japanese, and Vietnamese.
Preserve the original meaning, tone, line breaks, and formatting. Do not add explanations or notes.
Respond with a JSON object only, in exactly this shape:
{"english": "...", "japanese": "...", "vietnamese": "..."}"""

MSG_EMPTY_INPUT = "번역할 텍스트를 입력해주세요."
MSG_NO_API_KEY = "OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 Secrets를 확인하세요."
MSG_API_ERROR = "번역 중 오류가 발생했습니다: {}"
MSG_PARSE_ERROR = "번역 결과를 해석하지 못했습니다. 다시 시도해주세요."
MSG_INPUT_CHANGED = "입력한 글이 바뀌었습니다. 아래는 이전 글의 번역입니다. 다시 번역하려면 [번역하기]를 누르세요."


class TranslationParseError(Exception):
    pass


def load_config() -> dict:
    """Read settings from .env / env vars first, then st.secrets."""
    load_dotenv()

    def get(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        if value:
            return value
        try:
            return st.secrets.get(name, default)
        except Exception:  # no secrets.toml present
            return default

    return {
        "api_key": get("OPENAI_API_KEY"),
        "model": get("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
    }


def parse_translation(content: str | None) -> dict:
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as e:
        raise TranslationParseError from e
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), str) for key in LANGUAGES
    ):
        raise TranslationParseError
    return {key: data[key] for key in LANGUAGES}


def translate(text: str, config: dict) -> dict:
    """Translate into all three languages with a single API call."""
    client = OpenAI(
        api_key=config["api_key"],
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return parse_translation(response.choices[0].message.content)


CSS = """
<style>
/* Windows has no flag emoji glyphs; Noto Color Emoji supplies them. */
@import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
.block-container { max-width: 1200px; padding-top: 2.5rem; }
h1 { word-break: keep-all; }
@media (max-width: 640px) { h1 { font-size: 2rem !important; } }
.lang-title {
  font-size: 1.1rem; font-weight: 600; margin: 0 0 0.5rem;
  font-family: "Source Sans", sans-serif, "Noto Color Emoji";
}
.empty-state { color: #8a8f98; font-size: 0.9rem; margin: 0.25rem 0 0.75rem; }
[data-testid="stCode"] pre { white-space: pre-wrap; word-break: break-word; }
[data-testid="stCode"] code { font-family: "Source Sans", sans-serif; font-size: 0.95rem; line-height: 1.6; }
</style>
"""


def reset() -> None:
    st.session_state.input_text = ""
    st.session_state.pop("result", None)
    st.session_state.pop("result_source", None)
    st.session_state.pop("message", None)


def request_translation() -> None:
    # Runs before the rerun, so the button is already disabled while translating.
    st.session_state.pending = True


def run_translation(text: str, config: dict) -> None:
    """Translate and store either a result or a message in session state."""
    st.session_state.pending = False
    st.session_state.pop("message", None)
    if not text.strip():
        st.session_state.message = ("warning", MSG_EMPTY_INPUT)
        return
    if not config["api_key"]:
        st.session_state.message = ("error", MSG_NO_API_KEY)
        return
    try:
        with st.spinner("번역 중입니다..."):
            st.session_state.result = translate(text, config)
        st.session_state.result_source = text
        return
    except TranslationParseError:
        st.session_state.message = ("error", MSG_PARSE_ERROR)
    except Exception as e:
        st.session_state.message = ("error", MSG_API_ERROR.format(e))
    # Don't leave an older translation on screen next to the error.
    st.session_state.pop("result", None)


def render() -> None:
    st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("🌐 다국어 번역기")
    st.caption("입력한 글을 영어 · 일본어 · 베트남어로 번역합니다")

    config = load_config()
    text = st.text_area(
        "번역할 텍스트",
        key="input_text",
        height=180,
        # Also gives a live counter while focused. Streamlit silently rejects a paste
        # that would exceed the limit, so the limit is stated in the placeholder.
        max_chars=MAX_INPUT_CHARS,
        placeholder=(
            f"번역할 텍스트를 입력하세요... "
            f"(최대 {MAX_INPUT_CHARS:,}자, 언어는 자동으로 감지됩니다)"
        ),
    )

    busy = st.session_state.get("pending", False)
    col_translate, col_reset, _ = st.columns([1, 1, 6])
    col_translate.button(
        "번역하기",
        type="primary",
        width="stretch",
        disabled=busy,
        on_click=request_translation,
    )
    col_reset.button("초기화", width="stretch", disabled=busy, on_click=reset)

    if busy:
        run_translation(text, config)
        st.rerun()  # re-enable the buttons and render the outcome

    message = st.session_state.get("message")
    if message:
        level, body = message
        (st.warning if level == "warning" else st.error)(body)

    st.divider()
    result = st.session_state.get("result")
    if result and text != st.session_state.get("result_source"):
        st.info(MSG_INPUT_CHANGED)
    for column, (key, label) in zip(st.columns(3), LANGUAGES.items()):
        with column.container(border=True):
            st.markdown(f'<p class="lang-title">{label}</p>', unsafe_allow_html=True)
            if result:
                st.code(result[key], language=None, wrap_lines=True)
            else:
                st.markdown(
                    '<p class="empty-state">번역 결과가 여기에 표시됩니다.</p>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    render()
