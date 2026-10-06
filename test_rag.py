from rag import search_knowledge


question = "What factors affect loan approval?"

results = search_knowledge(question)


for i, result in enumerate(results):

    print("\nRESULT", i + 1)

    print(result.page_content)