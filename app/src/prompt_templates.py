import ollama
import langchain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class Parser(BaseModel):
    disinfo: bool = Field(description="Is given article disinformation")
    score: float = Field(description="Confidence score in [0,1]")
    output: str = Field(description="Short explanation (max 3 sentences)")

parser = JsonOutputParser(pydantic_object=Parser)
#fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
format_instructions = parser.get_format_instructions()

def zero_shot_prompt():
    template = """
    You are a fact-checking assistant. Classify the following article for **disinformation**.

    Definitions:
    - "disinformation": verifiably false or highly misleading claims presented as facts, often lacking credible sources, using fabricated data, misattributed quotes, or deceptive context.

    Decision heuristics (apply pragmatically):
    - Red flags: fabricated stats, conspiracy framing, sensational certainty without sources, miscaptioned media, authority impersonation, out-of-context evidence.
    - Green flags: multiple independent reputable sources, cautious language, transparent uncertainty, primary data links.

    Output must follow the JSON schema exactly.

    {format_instructions}

    Article:
    {input}
    """
    prompt_template = PromptTemplate(
        template=template.strip(),
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
        )
    #prompt_template.invoke({"input": question})
    return prompt_template

def one_shot_prompt():
    example_input = """Headline: "NASA confirms Earth will go dark for 15 days in November."
        Body: A viral post claims NASA announced a 'blackout' due to planetary alignment, citing fake quotes and no links."""
    example_output = {
                "disinfo": True,
                "score": 0.94,
                "output": "Classic recurring hoax: NASA has issued no such advisory; similar claims have been debunked repeatedly and provide no credible sources."
            }

    template = """
        You are a fact-checking assistant. Classify the following article for **disinformation** and reply **only** with JSON.

        Guidelines:
        - Be decisive but calibrated: use a score in [0,1] reflecting confidence.
        - Keep the explanation under 3 sentences; cite concrete reasons (fabricated claims, lack of sources, etc.).

        Example:
        Input:
        {example_input}

        Valid JSON Output:
        {example_output}

        Now analyze the new article.

        {format_instructions}

        Article:
        {input}
        """
    
    prompt_template = PromptTemplate(
        template=template.strip(),
        input_variables=["input"],
        partial_variables={
            "format_instructions": format_instructions,
            "example_input": example_input,
            "example_output": example_output,
        },
        )
    #prompt_template.invoke({"input": question})
    return prompt_template


def few_shot_prompt():
    example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

    examples = [
            {
                "question":
                """A blog asserts radio waves weaken immunity and 'activate' the virus; no peer-reviewed sources; misinterprets WHO statements.""",
                "answer": {
                    "disinfo": True,
                    "score": 0.98,
                    "output": "Claim is false and repeatedly debunked; 5G radio waves cannot create or 'activate' viruses and no credible studies support it."
                },
            },
            {
                "question":
                """A viral thread claims an official 'ban' exists; cites a cropped screenshot without document links; conflates climate targets with prohibition.""",
                "answer": {
                    "disinfo": True,
                    "score": 0.88,
                    "output": "Misrepresents policy: no EU-wide 'ban'; claim relies on a cropped image without provenance and incorrect interpretation of climate objectives."
                },
            },
            {
                "question":
                """Meeting minutes and two local outlets confirm discussion; quotes match the minutes; no extraordinary claims.""",
                "answer": {
                    "disinfo": False,
                    "score": 0.78,
                    "output": "Plausible, supported by multiple local sources and official minutes; no indicators of fabrication."
                },
            },
        ]

    prefix = (
        "Return a JSON object that strictly follows this schema.\n\n"
        "{format_instructions}\n\n"
        "Classify each new article without revealing your reasoning steps; only the short explanation in JSON."
    )

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="Article:\n{input}",
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
    )
    return prompt

def rag_prompt():
    prompt = PromptTemplate(
        template="template.strip()",
        input_variables=["input", "context"],
        partial_variables={"format_instructions": format_instructions,},
    )
    return prompt
