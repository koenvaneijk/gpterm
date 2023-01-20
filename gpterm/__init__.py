import os

from openai.error import AuthenticationError

from platformdirs import user_config_dir

from rich import print
from rich.console import Console
from rich.markdown import Markdown

from rich.prompt import Prompt
import json
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory

agents = {
    "Assistant": """\
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, \
from answering simple questions to providing in-depth explanations and \
discussions on a wide range of topics. As a language model, Assistant \
is able to generate human-like text based on the input it receives, \
allowing it to engage in natural-sounding conversations and provide \
responses that are coherent and relevant to the topic at hand.

Assistant is able \
to generate its own text based on the input it receives, allowing it to \
engage in discussions and provide explanations and descriptions on a \
wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range \
of tasks and provide valuable insights and information on a wide range \
of topics. Whether you need help with a specific question or just want \
to have a conversation about a particular topic, Assistant is here to \
assist.

Assistant always provides answers in GitHub-flavored Markdown.

{history}
Human: {human_input}
Assistant:""",
    "Python Expert": """
    Python Expert is a large language model trained by OpenAI specialized in \
    the Python programming language.
    
    Python Expert is designed to be able to assist with a wide range of tasks, \
    from answering simple questions to providing in-depth explanations and \
    discussions on Python. As a language model, Python Expert is able to \
    generate above human-level code suggestions and provide responses that \
    are coherent and relevant to the topic at hand.

    Python Expert is able to generate its own code suggestions based on the \
    input it receives, allowing it to engage in discussions and provide \
    explanations and descriptions on Python.

    Overall, Python Expert is a powerful tool that can help with a wide range \
    of tasks and provide valuable insights and information on Python. Whether \
    you need help with a specific question or just want to have a conversation \
    about Python, Python Expert is here to assist.

    Python Expert always provides answers in GitHub-flavored Markdown.

    {history}
    Human: {human_input}
    Python Expert:""",
}


class GPTerm:
    def __init__(self):
        self.agent = "Python Expert"
        self.template = agents[self.agent]

        self.config = self.load_config()

        if not self.config.get("openai_api_key"):
            self.setup_openai()

        os.environ["OPENAI_API_KEY"] = self.config["openai_api_key"]

        self.prompt = PromptTemplate(
            input_variables=["history", "human_input"], template=self.template
        )

        self.chain = LLMChain(
            llm=OpenAI(temperature=0),
            prompt=self.prompt,
            verbose=False,
            memory=ConversationalBufferWindowMemory(k=2),
        )

    def setup_openai(self):
        self.config["openai_api_key"] = Prompt.ask(
            "Enter your OpenAI API key (you will only have to do this once) \
        and you can get it [link=https://beta.openai.com/account/api-keys]here[/link]"
        )

    def load_config(self):
        # Determine config path
        config_dir = user_config_dir("gpterm", "gpterm")

        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)

        # Load config
        config_path = os.path.join(config_dir, "config.json")
        if os.path.exists(config_path):
            config = json.load(open(config_path))

        else:
            config = {}
            json.dump(config, open(config_path, "w"))

        return config

    def select_agent(self):
        print("""[bold]Select an agent[/bold]""")

    def reset_config(self):
        config_dir = user_config_dir("gpterm", "gpterm")
        config_path = os.path.join(config_dir, "config.json")

        if os.path.exists(config_path):
            os.remove(config_path)
            print("Config reset. Please restart GPTerm.")

    def run(self):
        print(
            f"""Welcome to [bold]GPTerm[/bold], a terminal interface for \
GPT-3. Select your agent with 'select'. Quit with Ctrl+C or typing 'quit' or 'exit'. Type 'help' for \
more information.
Current agent: [bold green]{self.agent}[/bold green] (default)"""
        )

        while True:
            input_ = Prompt.ask("[bold]Human[/bold]")

            if input_ in ["quit", "exit"]:
                break

            if input_ == "reset_config":
                self.reset_config()
                break

            if input_ == "setup_openai":
                self.setup_openai()
                continue

            if input_ == "select":
                self.select_agent()

            if input_ == "help":
                print(
                    "GPTerm is a terminal interface for GPT-3 similar to ChatGPT. It is similar \
to ChatGPT, but it is a terminal interface instead of a web interface. \
GPTerm is a work in progress, so it is not very feature-rich yet."
                )
                continue

            try:
                output = self.chain.predict(human_input=input_)

                console = Console()
                md = Markdown(output)
                console.print("[bold]Assistant:[/bold]")
                console.print(md)
                console.print("\n")

            except AuthenticationError:
                print("Authentication error. Please check your OpenAI API key.")
                self.config["openai_api_key"] = ""
                break
