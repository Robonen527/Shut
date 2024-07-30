import cohere
import os
import hnswlib
import json
import uuid
from typing import List, Dict

import torch
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from transformers import AutoTokenizer, BertForMaskedLM, AutoModel
from unstructured.partition.text import partition_text

co = cohere.Client('2ELFLZKqLyZi5bLIGwt1kDBMpoAT9ch44DHfycAm')  # This is your trial API key
model_name = 'dicta-il/BEREL_2.0'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained('dicta-il/BEREL_2.0')
model.eval()

def get_embeddings(texts):
    # Tokenize and convert to tensors
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden state (token embeddings)
    embeddings = outputs.last_hidden_state

    # Average the token embeddings to get sentence embeddings
    sentence_embeddings = torch.mean(embeddings, dim=1).tolist()

    return sentence_embeddings


class Documents:

    def __init__(self, sources: List[Dict[str, str]]):
        self.sources = sources
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 5
        self.load()
        self.embed()
        self.index()

    def load(self) -> None:
        """
        Loads the documents from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for source in self.sources:
            elements = partition_text(filename=source["fileName"], strategy="hi_res", include_page_breaks=True)
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": source["title"],
                        "text": str(chunk),
                        "fileName": source["fileName"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the documents using the BEREL2.0 model.
        """
        print("Embedding documents...")

        batch_size = 70
        self.docs_len = len(self.docs)

        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = get_embeddings(texts)
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.index = hnswlib.Index(space="ip", dim=768)
        self.index.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.index.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.index.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the given query.

        Parameters:
        query (str): The query to retrieve documents for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.
        """
        docs_retrieved = []

        # Get embeddings for the query using BEREL_2.0
        query_emb = get_embeddings([query])[0]

        # Perform retrieval using the indexed embeddings
        doc_ids = self.index.knn_query([query_emb], k=self.retrieve_top_k)[0][0]

        docs_to_rerank = [self.docs[doc_id]["text"] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-multilingual-v3.0",
        )

        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results]

        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "fileName": self.docs[doc_id]["fileName"],
                }
            )

        return docs_retrieved


class Chatbot:

    def __init__(self, docs: Documents):
        self.docs = docs
        self.conversation_id = str(uuid.uuid4())

    def generate_response(self, message: str):
        """
        Generates a response to the user's message.

        Parameters:
        message (str): The user's message.

        Yields:
        Event: A response event generated by the chatbot.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """

        # Generate search queries (if any)
        response = co.chat(message=message, model="command-r", search_queries_only=True)

        if response.search_queries:
            print("Retrieving information...")

            documents = self.retrieve_docs(response)
            response = co.chat(
                message=message,
                documents=documents,
                model="command-r",
                conversation_id=self.conversation_id,
                stream=True,
            )
            for event in response:
                yield event
            yield response

            # If there is no search query, directly respond
        else:
            print("the question was not about the bible, sorry.\nask a question about the bible!")
            return

            response = co.chat(
                message=message,
                conversation_id=self.conversation_id,
                stream=True
            )
            for event in response:
                yield event

    def retrieve_docs(self, response) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the search queries in the response.

        Parameters:
        response: The response object containing search queries.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Get the query(s)
        queries = []
        for search_query in response.search_queries:
            queries.append(search_query["text"])

        # Retrieve documents for each query
        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(self.docs.retrieve(query))

        return retrieved_docs


class App:
    def __init__(self, chatbot: Chatbot):
        """
        Initializes an instance of the App class.

        Parameters:
        chatbot (Chatbot): An instance of the Chatbot class.

        """
        self.chatbot = chatbot

    def run(self):
        """
        Runs the chatbot application.
        """
        while True:
            # Get the user message
            message = input("User: ")

            # Typing "quit" ends the conversation
            if message.lower() == "quit":
                print("Ending chat.")
                break
            else:
                print(f"User: {message}")

            response = self.chatbot.generate_response(message)

            # Print the chatbot response
            print("Chatbot:")
            citations_flag = False

            for event in response:
                stream_type = type(event).__name__

                # Text
                if stream_type == "StreamTextGeneration":
                    print(event.text, end="")

                # Citations
                if stream_type == "StreamCitationGeneration":
                    if not citations_flag:
                        print("\n\nCITATIONS:")
                        citations_flag = True
                    print(event.citations[0])

                # Documents
                if citations_flag:
                    if stream_type == "StreamingChat":
                        print("\n\nDOCUMENTS:")
                        documents = [{'id': doc['id'],
                                      'text': doc['text'][:50] + '...',
                                      'title': doc['title'],
                                      'fileName': doc['fileName']}
                                     for doc in event.documents]
                        for doc in documents:
                            print(doc)

            print(f"\n{'-' * 100}\n")


sources = []
for i in range(50):
    sources.append(
        {
            "title": "תנך - בראשית " + str(i + 1),
            "fileName": "bereshit\\bereshit " + str(i + 1) + ".txt"
        }
    )

documents = Documents(sources)

chatbot = Chatbot(documents)

app = App(chatbot)

app.run()
