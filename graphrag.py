from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    ConfigurableField
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase, Driver
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.document_loaders import WikipediaLoader
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

graph = Neo4jGraph()


loader = TextLoader(file_path="dummytext.txt")
# loader = PyPDFLoader(file_path="Cosa_Nostra_-_John_Dickie.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(documents=docs)

driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )

def clear_specific_source(tx):
    tx.run("""
        MATCH (n)
        WHERE n.source = 'dummytext.txt'
        DETACH DELETE n
    """)

def clear_database(tx):
    tx.run("""
        MATCH (n)
        DETACH DELETE n
    """)    

with driver.session() as session:
    session.execute_write(clear_specific_source)

graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
vector_retriever = vector_index.as_retriever()

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ..., 
        description="All entities (people, organizations, places) appearing in the text"
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract all entities (people, organizations, locations) from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

output = entity_chain.invoke({"question": "Who are Nonna Lucia?"})

# Extract the names attribute from the output
print(output.names)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

graph.query(
    "CREATE FULLTEXT INDEX `fulltext_entity_id` IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


print(structured_retriever("Who is Nonna Lucia"))

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data
	
# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOllama(model = llm_model, temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


chain.invoke({"question": "Who is most influencial in Amico’s Family"})

chain.invoke(
    {
        "question": "what are some of the examples",
        "chat_history": [("Who is most influencial in Amico’s Family?", "Amico's great-grandfather, Antonio Caruso, played a pivotal role in shaping the family's story and culinary style")],
    }
)