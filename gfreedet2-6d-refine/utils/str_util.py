import re


def safe_to_camel_case(text: str, upper_first: bool = False) -> str:
    if not isinstance(text, str):
        raise TypeError("input must be a string")
    if not text:
        return ""

    cleaned = re.sub(r"[-_]+", "_", text.strip())
    words = cleaned.split("_")

    def capitalize_word(w):
        return w.capitalize() if w.isupper() else w[0].upper() + w[1:].lower()

    if upper_first:
        return "".join(capitalize_word(w) for w in words)
    else:
        return words[0].lower() + "".join(capitalize_word(w) for w in words[1:])
