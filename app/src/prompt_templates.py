import ollama
import langchain
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class Parser(BaseModel):
    disinfo: bool = Field(description="Is given article disinformation")
    score: float = Field(description="Confidence score")
    output: str = Field(description="Short explanation")

parser = JsonOutputParser(pydantic_object=Parser)
format_instructions = parser.get_format_instructions()

def zero_shot_prompt():
    prompt_template = PromptTemplate(
        template="Is this article disinformation?\n{format_instructions}\n{input}\n",
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
        )
    #prompt_template.invoke({"input": question})
    return prompt_template

def one_shot_prompt():
    prompt_template = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{input}\n",
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
        )
    #prompt_template.invoke({"input": question})
    return prompt_template


def few_shot_prompt():
    example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

    examples = [
        {
            "question": "Who lived longer, Muhammad Ali or Alan Turing?",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: How old was Muhammad Ali when he died?
    Intermediate answer: Muhammad Ali was 74 years old when he died.
    Follow up: How old was Alan Turing when he died?
    Intermediate answer: Alan Turing was 41 years old when he died.
    So the final answer is: Muhammad Ali
    """,
        },
        {
            "question": "When was the founder of craigslist born?",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Who was the founder of craigslist?
    Intermediate answer: Craigslist was founded by Craig Newmark.
    Follow up: When was Craig Newmark born?
    Intermediate answer: Craig Newmark was born on December 6, 1952.
    So the final answer is: December 6, 1952
    """,
        },
        {
            "question": "Who was the maternal grandfather of George Washington?",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Who was the mother of George Washington?
    Intermediate answer: The mother of George Washington was Mary Ball Washington.
    Follow up: Who was the father of Mary Ball Washington?
    Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
    So the final answer is: Joseph Ball
    """,
        },
        {
            "question": "Are both the directors of Jaws and Casino Royale from the same country?",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Who is the director of Jaws?
    Intermediate Answer: The director of Jaws is Steven Spielberg.
    Follow up: Where is Steven Spielberg from?
    Intermediate Answer: The United States.
    Follow up: Who is the director of Casino Royale?
    Intermediate Answer: The director of Casino Royale is Martin Campbell.
    Follow up: Where is Martin Campbell from?
    Intermediate Answer: New Zealand.
    So the final answer is: No
    """,
        },
    ]

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=(
            "Return a JSON object that strictly follows this schema.\n\n"
            "{format_instructions}\n\n"),
        suffix="Question: {input}",
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
    )
    #print(prompt.invoke({"input": question}).to_string())
    return prompt



#chain = prompt | llm | parser

#chain.invoke({"input": joke_query})