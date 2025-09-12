from abc import ABC
from .core import Agent
from utilities import PROMPTS, format_prompt


class ResponseAgent(Agent, ABC):
    """
    An agent that generates natural language responses to user questions based on SQL query results.

    This agent acts as a data assistant, interpreting SQL query results and transforming them into user-friendly, natural language answers. It utilizes a language model (currently Gemini) to craft responses that effectively convey the information derived from the data.

    Attributes:
        agentType (str): Indicates the type of agent, fixed as "ResponseAgent".

    Methods:
        run(user_question, sql_result) -> str:
            Generates a natural language response to the user's question based on the SQL query result.

            Args:
                user_question (str): The question asked by the user in natural language.
                sql_result (str): The result of the SQL query executed to answer the question.

            Returns:
                str: The generated natural language response.
    """


    agentType: str = "ResponseAgent"

    def run(self, user_question, sql_result):

        context_prompt = PROMPTS['nl_reponse']



        context_prompt = format_prompt(context_prompt,
                                       user_question = user_question,
                                       sql_result = sql_result)
                                       
        print(f"Prompt for Natural Language Response: \n{context_prompt}")

        generated_sql = self.generate_llm_response(context_prompt)
        return generated_sql

    