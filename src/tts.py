import pyttsx3


def speak(text: str) -> None:
    """
    Speak *text* aloud using the system's default TTS voice.

    A new ``pyttsx3.init()`` engine is created on each call so
    that the function is safe to use from different threads
    (e.g. inside a Streamlit button callback).  If the audio
    backend is unavailable the exception is swallowed silently
    so the host application never crashes.

    Parameters
    ----------
    text : str
        The string to convert to speech.

    Returns
    -------
    None
    """
    if not text or not text.strip():
        return

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)       # words per minute
        engine.setProperty("volume", 1.0)     # 0.0 – 1.0
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        # Silently ignore TTS failures — app must stay alive
        pass