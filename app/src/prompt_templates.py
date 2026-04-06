import json
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class Parser(BaseModel):
    disinfo: bool = Field(description="Is given article disinformation")
    score: float = Field(description="Confidence score in [0,1]")
    output: str = Field(description="Short explanation (max 3 sentences)")

parser = JsonOutputParser(pydantic_object=Parser)
format_instructions = parser.get_format_instructions()

def zero_shot_prompt():
    template = """
    You are a fact-checking assistant. Classify the following article for **disinformation**.

    Definitions:
    - "disinformation": verifiably false or highly misleading claims presented as facts, often lacking credible sources, using fabricated data, misattributed quotes, or deceptive context.

    Decision heuristics (apply pragmatically):
    - Red flags: fabricated stats, conspiracy framing, sensational certainty without sources, miscaptioned media, authority impersonation, out-of-context evidence.
    - Green flags: multiple independent reputable sources, cautious language, transparent uncertainty, primary data links.

    Return strictly json output like this:

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
    example_input = """It s only been a couple days since Republicans received massive backlash for trying to gut the Office of Congressional Ethics. Not they are doing something just as despicable. Republicans hate federal workers and Donald Trump s team has been leading a witch hunt demanding names of federal workers who work on climate change and LGBT issues. Clearly, they have something sinister in mind by specifically targeting those federal employees. And under the cover of the uproar over Republican efforts to kill the ethics panel, Republicans passed a rule known as the Holman Rule that gives Republicans the power to target specific individual federal workers and programs. According to the Washington Post,The rule changes the process of passing spending bills by allowing any rank-and-file House member to propose an amendment that would cut a specific federal program or the jobs of specific federal employees, by slashing their salaries or eliminating their positions altogether. Before this rule change, an agency s budget could be cut broadly, but a specific program, employee or groups of employees could not be targeted because of civil service protections. For example, Republicans could use the rule to cut a specific federal worker s pay down to a $1. Yeah, you read that right. One whole dollar. So let s say Republicans find out that a certain federal worker in the Energy Department in studying the effects of climate change. They could persecute that person by cutting their pay to force them to stop doing their job. Or they could cut the pay of a federal land management employee as punishment for protecting public lands from hooligans like the Bundy brothers. It s an absolutely terrifying rule that makes it incredibly clear that Republicans intend to destroy anyone who doesn t agree with them. This is tyranny, pure and simple.Maureen Gilman, who represents 150 federal employees as legislative director for the National Treasury Employees Union, is very concerned. This is part of a very chilling theme that federal workers are seeing right now,  Gilman told the Washington Post.House Minority Whip Steny Hoyer criticized the rule on Tuesday as  shortsighted and ideologically driven changes to our civil service  in step with Republican efforts to use federal employees as scapegoats.Rep. H. Morgan Griffith is the Republican who revived the rule and his opinion on the rule is even more frightening because he compared it to using a sniper rifle against the federal government and admitted that Republicans could use the law to send thousands of federal employees to the unemployment line.He favors a strategic application of the law, likening it to a bullet from a sniper rifle rather than a shotgun. It s unlikely   but not impossible   that members will  go crazy  and cut huge swaths of the workforce, he said. I can t tell you it won t happen,  he said in an interview in his office.  The power is there. But isn t that appropriate?This is yet another outrageous rule Republicans are using to attack the government they despise. It s a way for them to launch personal vendettas and get revenge and anybody who doesn t agree with their extremist ideology is in danger of losing their livelihood. It s an attack that should enrage the American people."""
    example_output = {"disinfo": True, "score": 0.9, "output": "The article contains miscaptioned media (the Washington Post is not directly quoted), and out-of-context evidence. The article also relies on a single source for the Holman Rule's details, which may be incomplete or inaccurate. Additionally, the article's language is emotive and biased, with phrases like 'tyranny, pure and simple' and 'extremist ideology'."}


    template = """
        You are a fact-checking assistant. Classify the following article for **disinformation** and reply **only** with JSON.

        Definitions:
        - "disinformation": verifiably false or highly misleading claims presented as facts, often lacking credible sources, using fabricated data, misattributed quotes, or deceptive context.

        Guidelines:
        - Red flags: fabricated stats, conspiracy framing, sensational certainty without sources, miscaptioned media, authority impersonation, out-of-context evidence.
        - Green flags: multiple independent reputable sources, cautious language, transparent uncertainty, primary data links.
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



def _escape_braces(s: str) -> str:
    """Escapes braces so .format() won't treat them as placeholders."""
    return s.replace("{", "{{").replace("}", "}}")


def few_shot_prompt():
    example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

    examples = [
            {
                "question":
                """WATCH: Paul Ryan Just Told Us He Doesn’t Care About Struggling Families Living In Blue States Republicans are working overtime trying to sell their scam of a tax bill to the public as something that directly targets middle-class and working-class families with financial relief. Nothing could be further from the truth, and they re getting hammered on that repeatedly. Featured image via Mark Wilson/Getty Images.""",
                "answer": {"disinfo": True, "score": 0.8, "output": "The article contains several red flags, including sensational certainty without sources, miscaptioned media (Chip Somodevilla/Getty Images), and conspiracy framing (e.g., 'He knows Donald Trump is destroying the GOP'). While it cites a reputable source (NPR) for some quotes, it selectively presents information to create a misleading narrative."},
            },
            {
                "question":
                """Financial firms fear turmoil over fraught U.S. debt ceiling talks Financial firms are sounding alarm bells and dusting off contingency plans over fears an increasingly dysfunctional U.S. Congress may fail to reach a deal to raise the country’s debt limit. Several lobbyists, representing dozens of bankers, investors and credit rating agencies, told Reuters they are worried that dynamics at play in Washington –  a bitterly divided Republican party and unpredictable President Donald Trump – could rule out a deal before an October deadline. """,
                "answer": {"disinfo": False, "score": 0.8, "output": "This article appears to be factual and based on credible sources, including Reuters and statements from industry executives and government officials. However, some language used by lobbyists and analysts may be sensational or alarmist, but it does not appear to be intentionally misleading."},
            },
            {
                "question":
                """Former Attorney General Eric Holder responded to a report that Donald Trump asked Attorney General Jeff Sessions to drop the charges against Joe Arpaio (aka America s most racist sheriff) by throwing shade at the crooked administration on Twitter.The Washington Post published an article on Saturday claiming that Trump had personally asked Sessions to abandon the case against Arpaio, who was convicted of criminal contempt charges after ignoring a federal court order. Holder tweeted out that in all the years he had worked as Attorney General under former President Obama, he had never once been asked to drop charges against anybody, ever. """,
                "answer": {"disinfo": True, "score": 0.6, "output": "The article contains claims about Eric Holder's assertion that President Obama never asked him to drop cases, which lacks independent verification and appears to be a fabricated or exaggerated statement. While the Washington Post is a reputable source, the specific claim about Obama's calls is not corroborated by public records. The article also presents Trump's actions and Arpaio's conviction as factual, but the framing of Holder's statement as an absolute fact without context creates misleading certainty."},
            },
        ]

    formatted_examples = [
        {
            "question": ex["question"],
            "answer": _escape_braces(json.dumps(ex["answer"], ensure_ascii=False))
        }
        for ex in examples
    ]

    example_prompt = PromptTemplate.from_template(
        "Example Article:\n{question}\n\nExample JSON Output:\n{answer}\n"
    )

    prefix = """
    You are a fact-checking assistant.
    Classify the article for **disinformation** and return ONLY valid JSON.

    {format_instructions}

    Here are examples:
    """

    prompt = FewShotPromptTemplate(
        examples=formatted_examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="Article:\n{input}",
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
    )

    return prompt



def rag_prompt():
    template = """
    Inputs:
    - context: a list of short texts that are KNOWN examples of disinformation / fake news from this dataset.
    - article: the news article to classify.

    Treat the context texts as POSITIVE EXAMPLES of fake news from this dataset. They show the typical tone, style, and purpose that the dataset labels as fake.

    Your job :
    Decide if the article is disinformation according to BOTH:
    1) general principles of disinformation, and
    2) similarity to the context examples (style, tone, and intent), which are known fake from this dataset.
    
    Rules:
    Similarity to context (RAG usage)
    - Compare the article to the context examples.
    - If the article is semantically similar to them in topic, tone, and style (e. g., emotional attacks, partisan rhetoric, insults, mockery, strong bias), and it is not explicitly labeled as opinion lean toward disinformation.
    - If the article is clearly analyzing, fact-checking, or criticizing the patterns you see in the context examples lean toward not_disinformation.

    Return strictly JSON using this format:

    {format_instructions}

    Context (disinformation tweets):
    {context}

    Article:
    {input}
    """

    prompt = PromptTemplate(
        template=template.strip(),
        input_variables=["input", "context"],
        partial_variables={"format_instructions": format_instructions,},
    )
    return prompt
