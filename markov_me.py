from pathlib import Path
from functools import lru_cache

import markovify


def generate_sentence():
    """Generate excellent sentences forever."""
    text_model = load_data()
    while True:
        yield text_model.make_sentence()


@lru_cache(maxsize=1)
def load_data():
    # Load up subtitle files that have been scrubbed to only have dialogue.
    sources = Path('./sources/sanitized').glob('**/*.srt')
    text = ''
    for source in sources:
        with source.open() as file:
            text += file.read()

    text_model = markovify.Text(text)
    return text_model


def main():
    for i in generate_sentence():
        print(i)
        input()


if __name__ == '__main__':
    main()
