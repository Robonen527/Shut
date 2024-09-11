from typing import List, Dict

import cohere
import hnswlib
import uuid
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.text import partition_text

co = cohere.Client('2ELFLZKqLyZi5bLIGwt1kDBMpoAT9ch44DHfycAm')  # This is your trial API key

class Documents:
    # loads the documents, the embed and index happens here.
    def __init__(self, sources: List[Dict[str, str]]):
        self.sources = sources
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 5

    def lei(self, index_dim):
        self.load()
        self.embed()
        self.index(index_dim)

# Loads the documents from the sources and chunks the HTML content.
    def load(self) -> None:
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

    # Embeds the documents using the Cohere API.
    def embed(self) -> None:
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

    # Indexes the documents for efficient retrieval.
    def index(self, dim) -> None:
        print("Indexing documents...")

        self.index = hnswlib.Index(space="ip", dim=dim)
        self.index.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.index.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.index.get_current_count()} documents.")

    """-------------------------------------------
    Retrieves documents based on the given query.

    Parameters:
    query (str): The query to retrieve documents for.

    Returns:
    List[Dict[str, str]]: A list of dictionaries representing the retrieved
    documents, with 'title', 'snippet', and 'url' keys.
    ----------------------------------------------"""
    def retrieve(self, query: str) -> List[Dict[str, str]]:
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
                    "fileName": self.docs[doc_id]["fileName"],
                }
            )

        return docs_retrieved


class Chatbot:
    def __init__(self, docs: Documents):
        self.docs = docs
        self.conversation_id = str(uuid.uuid4())

    """------------------------------------------
    Generates a response to the user's message.

    Parameters:
    message (str): The user's message.

    Yields:
    Event: A response event generated by the chatbot.

    Returns:
    List[Dict[str, str]]: A list of dictionaries representing the retrieved
    documents.
    -------------------------------------------"""
    def generate_response(self, message: str):
        # Generate search queries (if any)
        response = co.chat(message=message, model="command-r-plus", search_queries_only=True)

        if response.search_queries:
            print("Retrieving information...")

            documents = self.retrieve_docs(response)
            response = co.chat(
                message=message,
                documents=documents,
                model="command-r-plus",
                conversation_id=self.conversation_id,
                stream=True,

            )
            for event in response:
                yield event
            if not response.documents:
                print("השאלה לא הייתה קשורה לתנך, סליחה\nשאל שאלה על התנך!")
                return
            yield response

        # If there is no search query, directly respond
        else:
            print("the question was not about the bible, sorry.\nask a question about the bible!")
            return

    """----------------------------------------------------------
    Retrieves documents based on the search queries in the response.

    Parameters:
    response: The response object containing search queries.

    Returns:
    List[Dict[str, str]]: A list of dictionaries representing the retrieved
    documents.
    ----------------------------------------------------------"""
    def retrieve_docs(self, response) -> List[Dict[str, str]]:
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
        self.chatbot = chatbot

    # Prints the documents' title.
    def print_mekorot(self, documents):
        str = ''
        for document in documents:
            str += document['title'] + ", "
        print("תוכל למצוא את התשובה ב:\n" + str)

    # Runs the chatbot application.
    def run(self):
        while True:
            # Gets the user message
            message = input("User: ")

            # Typing "quit" ends the conversation
            if message.lower() == "quit":
                print("Ending chat.")
                break

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
                        self.printMekorot(documents)
                        for doc in documents:
                            print(doc)

            print(f"\n{'-' * 100}\n")


sources = []
for i in range(50):
    sources.append(
        {
            "title": "bereshit " + str(i + 1),
            "fileName": "bereshit\\bereshit " + str(i + 1) + ".txt"
        }
    )
# sources.append({"title": "bereshit",
#                 "fileName": "bereshit\\ereshit.txt"})

documents = Documents(sources)
documents.lei(1024)
chatbot = Chatbot(documents)
app = App(chatbot)
app.run()
