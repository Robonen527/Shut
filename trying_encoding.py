import chardet
import cohere
import os
import hnswlib
import json
import uuid
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

co = cohere.Client('2ELFLZKqLyZi5bLIGwt1kDBMpoAT9ch44DHfycAm')  # This is your trial API key


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

    def partition_html(self, url):
        response = requests.get(url)
        response.encoding = 'UTF-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup

    def chunk_by_title(self, soup):
        chunks = []
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            title = header.get_text(strip=True)
            content = ''
            for sibling in header.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    break
                content += sibling.get_text(strip=True) + '\n'
            chunks.append((title, content))
        return chunks

    def load(self):
        print("Loading documents...")
        print("\n\n זה קובץ שמקודד נורמאלי והטקסט שלו מתקבל טוב\n\n")
        for source in self.sources:
            soup = self.partition_html(url=source["url"])
            chunks = self.chunk_by_title(soup)
            for title, text in chunks:
                print(title, text)
                self.docs.append({
                    "title": title,
                    "text": text,
                    "url": source["url"],
                })
                print({
                    "title": title,
                    "text": text,
                    "url": source["url"],
                })
                exit(1)  # יציאה אחרי הדפסה ראשונה לבדיקת תקינות

    # def load(self):
    #     print("Loading documents...")
    #     for source in self.sources:
    #         soup = partition_html(url=source["url"])
    #         chunks = chunk_by_title(soup)
    #         for chunk in chunks:
    #             print(chunk)
    #             self.docs.append({
    #                 "title": source["title"],
    #                 "text": chunk,
    #                 "url": source["url"],
    #             })
    #             print({
    #                 "title": source["title"],
    #                 "text": chunk,
    #                 "url": source["url"],
    #             })
    #             exit(1)

    def embed(self) -> None:
        """
            Embeds the documents using the Cohere API.
            """
        print("Embedding documents...")

        batch_size = 90
        self.docs_len = len(self.docs)

        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
    Indexes the documents for efficient retrieval.
    """
        print("Indexing documents...")

        self.index = hnswlib.Index(space="ip", dim=1024)
        self.index.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.index.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.index.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
    Retrieves documents based on the given query.

    Parameters:
    query (str): The query to retrieve documents for.

    Returns:
    List[Dict[str, str]]: A list of dictionaries representing the retrieved  documents, with 'title', 'snippet', and 'url' keys.
    """

        docs_retrieved = []
        query_emb = co.embed(
            texts=[query],
            model="embed-multilingual-v3.0",
            input_type="search_query"
        ).embeddings

        doc_ids = self.index.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        docs_to_rerank = []
        for doc_id in doc_ids:
            docs_to_rerank.append(self.docs[doc_id]["text"])

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-multilingual-v3.0",
        )

        doc_ids_reranked = []
        for result in rerank_results:
            doc_ids_reranked.append(doc_ids[result.index])

        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
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
                                      'url': doc['url']}
                                     for doc in event.documents]
                        for doc in documents:
                            print(doc)

            print(f"\n{'-' * 100}\n")


sources = []
for i in range(50):
    sources.append(
        {
            "title": "תנך - בראשית " + str(1 + i),
            "url": "https://mechon-mamre.org/i/t/x/x0" + str(101 + i) + ".htm"
        }
    )

documents = Documents(sources)

chatbot = Chatbot(documents)

app = App(chatbot)

app.run()
