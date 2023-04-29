# Import the necessary libraries
import os
from gpt_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext, SimpleDirectoryReader
from langchain import OpenAI
import requests
import json
from bs4 import BeautifulSoup
import streamlit

os.environ["OPENAI_API_KEY"] = streamlit.secrets["openai_secret"]


# Define a class for downloading articles from Zendesk
class ZendeskArticlesLoader:
    def __init__(self):
        # Initialize the class with a Zendesk API token
        self.zendesk_token = "PFFlXg3GKoHvfWusCmeNV4sLPQMvVzBb3p1U9ndj"

    def save_articles_to_txt(self, url):
        """
        Downloads articles from a Zendesk Help Center API and saves them as plain text files.
        The articles are saved in the 'contents' directory with the article ID as the file name.
        Each article is saved with a source URL and article ID as a footer.

        Args:
            url (str): The URL of the Zendesk API endpoint for articles in the desired language.

        Returns:
            None
        """
        
        # Set the necessary headers for making requests to the Zendesk API
        headers = {
            "Authorization": self.zendesk_token,
            "Content-Type": "application/json"
        }

        # Make a GET request to the specified API endpoint
        response = requests.get(url, headers=headers)
        # print(response.content)

        # Load the articles from the response JSON into a dictionary
        articles = json.loads(response.content)

        # Iterate over each article in the response and save it as a plain text file
        for article in articles["articles"]:
            # Use BeautifulSoup to remove HTML tags from the article body
            soup = BeautifulSoup(article["body"], 'html.parser')
            text_without_tags = soup.get_text()
            # Add a footer to the plain text article with the source URL and article ID
            text_without_tags += f"\n\nsource: {article['html_url']} \narticle id = {str(article['id'])}"
            # print(text_without_tags)

            # Remove leading/trailing whitespace from each line of the plain text article
            lines = (line.strip() for line in text_without_tags.splitlines())
            # Combine the lines into a single string with each line separated by a newline character
            new_lines = '\n'.join(line for line in lines if line)

            # Save the plain text article to a file with the article ID as the file name
            with open(f"contents/{article['id']}.txt", "w") as f:
                f.write(new_lines)

        # Check if there are more pages of articles to download
        next_url = articles["next_page"]
        if next_url != None:
            # If so, recursively call the save_articles_to_txt method with the URL of the next page
            self.save_articles_to_txt(next_url)

    def construct_index(self, directory_path):
        """
        Constructs a GPT-based vector index from the text documents in the specified directory.

        Parameters:
        directory_path (str): The path to the directory containing the text documents.

        Returns:
        GPTSimpleVectorIndex: A GPT-based vector index instance constructed from the text documents.
        """
         
        # Set the parameters for the prompt helper
        max_input_size = 4096
        num_outputs = 256
        max_chunk_overlab = 20
        chunk_size_limit = 600

        # Create a PromptHelper instance with the given parameters
        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlab, chunk_size_limit)

        # Create an OpenAI language model instance with the specified configuration
        llm = OpenAI(temperature=0, model_name="text-davinci-002", max_tokens=num_outputs)

        # Create an LLMPredictor instance with the language model instance
        llm_predictor = LLMPredictor(llm)

        # Load the documents from the specified directory using the SimpleDirectoryReader class
        documents = SimpleDirectoryReader(directory_path).load_data()

        # Create a ServiceContext instance with the LLMPredictor instance
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

        # Create a GPTSimpleVectorIndex instance with the documents and ServiceContext instances
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

        # Save the index to disk in JSON format
        index.save_to_disk("indexes/zendesk_index.json")

        # Return the index instance
        return index

# Create an instance of the ZendeskArticlesLoader class
zal = ZendeskArticlesLoader()
# Call the save_articles_to_txt method with the URL of the Zendesk API endpoint for English-language articles
# zal.save_articles_to_txt("https://details.zendesk.com/api/v2/help_center/en-us/articles.json")

# Call the construct_index method of the ZendeskArticlesLoader instance, passing in the "contents" directory path
zal.construct_index("contents")
