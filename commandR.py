import cohere

co = cohere.Client('2ELFLZKqLyZi5bLIGwt1kDBMpoAT9ch44DHfycAm')
chat_history = []
while True:
    message = input()
    response = co.chat(
        chat_history=chat_history, message=message,
    # perform web search before answering the question. You can also use your own custom connector.
    connectors=[{"id": "web-search"}]
    )
    text = response.text
    chat_history += ([{"role": "USER", "message": message}, {"role": "CHATBOT", "message": text}])
    print(response.text[::-1])
    citations = response.citations
    if citations:
        print("citations: " + response.citations)