import setuptools

setuptools.setup(
    name="gpterm",
    version="0.0.1",
    author="Koen van Eijk",
    author_email="vaneijk.koen@gmail.com",
    description="A package for interacting with GPT-3 through terminal, similar to ChatGPT.",
    entry_points={"console_scripts": ["gpterm = gpterm.command_line:main"]},
    install_requires=["langchain", "openai"],
)
