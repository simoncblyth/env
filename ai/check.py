import openai
import os

#from dotenv import load_dotenv, find_dotenv
#_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')


def q(prompt, model="gpt-3.5-turbo"):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]







