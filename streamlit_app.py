__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from openai import OpenAI
import os
import bs4
from getpass import getpass
from typing import Sequence
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.docstore.document import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import requests
import asyncio
import nest_asyncio
import uuid 
import chromadb
from duckduckgo_search import DDGS
import subprocess
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

chromadb.api.client.SharedSystemClient.clear_system_cache()
# Ensure Playwright is properly set up (downloads necessary browsers)
try:
    subprocess.check_call(["playwright", "install"])
except Exception as e:
    print("Failed to install Playwright browsers:", e)

### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def scrape_website(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)  # Wait for network to be idle
        await page.wait_for_timeout(2000)
        html_content = await page.content()
        await browser.close()

        soup = BeautifulSoup(html_content, "html.parser")

        return soup  # Return the extracted data

def find_port_id(destination):
    words = destination.strip().lower().split(",")
    formatted_dest = destination.lower().replace(", ", "+")
    
    try:
        search_term = "Holland America.com " + destination

        ddgs = DDGS(timeout=20) 
        try:
            response = ddgs.text(search_term, max_results=3)
        except RatelimitException:
            return 'not found', ''
        first_link = response[0]['href']
    
        if all(word in first_link.lower() for word in words):
            port_code = first_link[-3:]
            exc_url = "https://www.hollandamerica.com/en/us/excursions?fq=portID:" + port_code
        else:
            port_code = 'not found'
            exc_url = ''
    
    except IndexError:
        port_code = 'not found'
        exc_url = ''
    
    return port_code, exc_url

def get_port_info(destination, vectorstore):
  
    words = destination.strip().lower().split(",")
    search_term = destination + "cruise port whatsinport"

    ddgs = DDGS(timeout=20) 
    try:
        response = ddgs.text(search_term, max_results=3)
    except RatelimitException:
        return False
    first_result = response[0]['href']

    
    print(response)

    # If port page found
    if (words[0].lower() in first_result.lower() ) and ('whatsinport' in first_result.lower()):

        wiki_flag = True
        response = requests.get(
        first_result,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
        )

        # Parse the results
        soup = BeautifulSoup(response.text, 'html.parser')
        port_info = soup.get_text(strip=True)
        print('got port info')

        port_docs= [Document(page_content=port_info, metadata={'source': first_result})]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks  =  text_splitter.split_documents(port_docs)
        
        vectorstore.add_documents(documents=chunks)
    else:
        wiki_flag = False
    
    return wiki_flag

def scrape_holland(port_code, exc_url, vectorstore):
    if port_code != 'not found':

        # Load, chunk and index the contents of the blog.
        exc_soup = asyncio.run(scrape_website(exc_url))
        docs = exc_soup.get_text(strip=True)
        docs= [Document(page_content=docs, metadata={'source': exc_url})]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks  =  text_splitter.split_documents(docs)

        vectorstore.add_documents(documents=chunks)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    return retriever

@st.cache_resource
def initialize_rag():
    """Initialize RAG components and return reusable objects"""

    st.session_state.first_run = True

    # Initialize core components
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)
    llm = ChatOpenAI(model="ft:gpt-4o-2024-08-06:personal:script-test3:AOYaYI9l")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="port_profiles",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    

    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # Set up prompts
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    return {
        "llm": llm,
        "vectorstore": vectorstore,
        "contextualize_q_prompt": contextualize_q_prompt,
        "qa_prompt": qa_prompt
    }

@st.cache_resource
def initialize_agent():
    # Create the workflow
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    
    memory = MemorySaver()
    agent = workflow.compile(checkpointer=memory)
    return agent



template = """
You are a copywriting assistant helping write scripts about popular cruise destinations, for the requested destination. Provide specific and unique descriptions for each destination, avoiding generic or broad statements. Avoid referencing specific individuals (like tour guides, local experts, chefs) due to potential changes in their availability. Historical figures are an exception to this rule. Refrain from mentioning specific local businesses (e.g., restaurants, cafes, bars). Check sailing schedules for each region and refrain from mentioning seasons with no active cruise operation. Use "we recommend" instead of "it's recommended" to convey a more personal and collective suggestion. Always address guests in the second person to maintain a personal and engaging tone. Here follows the template for an output of a script, where you see brackets of different sorts ()<>that means it’s a guidance and shouldn’t be included in the output, blanks are for you to fill in:

***
Welcome to Port Profiles, your quick guide to (City Name), (Country).

(Intro – 75 words)
 <Geographical location - like this city is located in this region, this coast, why that’s interesting - 1 sentence>. <Interesting point about the city/region like what it’s famous for - 1 sentence>. <Hook or teaser for major points of interests in the destination/region, for instance, take 3 major points of interests and talk about each one, comma separated, tie it all together - 1 big sentence>

We'll get into all this and more, but first, let's cover the basics.


(Port Info)

Docking Information (50 words):
<depending if the port is tender or not Pick one of the options>
Option #1: Our ship will come alongside the (Name of Cruise Port/Terminal) _________, conveniently located (distance – Example: less than a mile) __________ from the city/town center.
Option #2:
(Name of city/town) __________ is a tender port – meaning the city/town does not have a dock or pier large enough for our ship to come alongside.  Therefore, we will anchor offshore and shuttle guests to the pier, using our ship's tender boats.  Information regarding our specific tending operation will be addressed by your Cruise & Travel Director and printed in the Daily Program.
Weather (50 words):
 <Talk about the type of Koppen climate in the region, what kind of weather is there usually in summer/winter depending on the cruising season. Make a recommendation on how to dress, and if there are any other things to know like making sure to bring sunscreen>

Transportation (50 words):

If you’re looking for transportation, you'll find plenty of options near the passenger terminal, including taxis, <research what transportation options are available close to cruise terminal like bus, tram, ferry, etc> ______ &,ript Model P  For those booked on a Holland America Shore Excursion, check your Navigator App for your digital ticket, including your meeting location and time.  It's a hassle-free way to maximize your time in port.

Port/Visitor Information, Wi-Fi (50 words):

<Research if there is a visitor or tourist information center at the cruise terminal or nearby> Before heading out, make sure to visit the information (center/booth) ________ locatedModel Prompt
  For those looking to stay connected, free Wi-Fi is available at our docking location, along with many cafes and restaurants thought the city/town (make sure Wi-Fi information is correct, research it first).

Currency, ATM, Credit Cards (25 words):

The currency of (destination)__________ is the ____.  Credit and debit cards are widely accepted, and ATMs are available throughout the island/city/town for those looking for cash.

Language (25 words):

The official language of (destination)________ is ___, <add how widely spoken English is there>.

Now that we've covered some of the basics let's look at some highlights and options for how to spend your time ashore.

(Main Body – 3 Big Points of Interest with potentially some smaller supporting ones in the vicinity. This section should be around 300-350 word count)

<1st paragraph - Start with POI nearest to the docking location and work your way out. 4-5 sentences or about 80-100 words. Include 1 big POI and maybe some small supporting ones>

<2nd paragraph. Spread out from the city if there’s anything interesting in the general vicinity in the region, if the drive isn’t more than 3-4 hours. 4-5 sentences or about 80-100 words. Include 1 big POI and maybe some small supporting ones>

<3rd paragraph. Final big points of interest. 4-5 sentences or about 80-100 words. Include 1 big POI and maybe some small supporting ones>

(Food & Drink – 50 words)
<Highlight one dish tip.>
Example: If you find yourself getting hungry while exploring the city, don't miss the chance to try ___, a local favorite that's sure to satisfy your appetite

(Fun Fact – 75 words)
<Dig deep and find something truly spectacular. Avoid food facts>
***

Use the following pieces of context to answer the question at the end. Don't mention the prices of Shore Excursions, or their names explicitly, but talk about Points of Interests that are included in the excursions.
If you don't know the answer or there is some information you're not sure, just say that you don't know, don't try to make up an answer.


{context}

"""

def main():

    if 'first_run' not in st.session_state:
            st.session_state['first_run'] = True
            id = uuid.uuid1() 
            id = str(id.int)
            st.session_state['chat_id'] = id
            print(id)
            logging.info("generated chat id " + id)


    # Show title and description.
    st.title("Port Profiles Script Writing Agent")
    st.write(
        """
            Instructions:
                \n- For the first query just enter your destination in the format "city, country" (e.g. "Horta, Portugal")
                \n- You can ask for changes to the script in consecutive queries
                \n- If you want to start on a script for another destination - refresh the page
                \n- If something goes wrong refresh the page or reach out
        """
    )

    # Ask user for their OpenAI API key via `st.text_input`.
    # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
    # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
    global PERSIST_DIR
    PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

    rag_components = initialize_rag()

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("Enter your destination"):
        
        # Store and display the current prompt.    
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)


        if st.session_state.first_run:
            # Get port data
            
            with st.chat_message("assistant"):
                    st.write("Sure! Let me see what I can do..")
        
            st.session_state.messages.append({"role": "assistant", "content": "Sure! Let me see what I can do.."})

            port_info_flag = get_port_info(prompt, rag_components["vectorstore"])
            
            if port_info_flag:
                with st.chat_message("assistant"):
                    st.write("Found port info on wiki")
        
                st.session_state.messages.append({"role": "assistant", "content": "Found port info on wiki"})
            else:
                with st.chat_message("assistant"):
                    st.write("Didn't find port info on wiki")
        
                st.session_state.messages.append({"role": "assistant", "content": "Didn't find port info on wiki"})
        
            port_code, exc_url = find_port_id(prompt)
            
            if port_code != 'not found':
                with st.chat_message("assistant"):
                        st.write("Found port page on Holland\'s website")
            
                st.session_state.messages.append({"role": "assistant", "content": "Found port page on Holland\'s website"})
            else:
                with st.chat_message("assistant"):
                        st.write("Didn't find port page on Holland\'s website")
            
                st.session_state.messages.append({"role": "assistant", "content": "Didn't find port page on Holland\'s website"})
                
            # Get retriever with vectorstore
            retriever = scrape_holland(port_code, exc_url, rag_components["vectorstore"])
            
            # Create retrieval chain
            history_aware_retriever = create_history_aware_retriever(
                rag_components["llm"], 
                retriever, 
                rag_components["contextualize_q_prompt"]
            )
            
            global rag_chain  # Make available to call_model
            rag_chain = create_retrieval_chain(
                history_aware_retriever, 
                create_stuff_documents_chain(
                    rag_components["llm"], 
                    rag_components["qa_prompt"]
                )
            )

            base_query = "Write me a script about {}".format(prompt)

            config = {"configurable": {"thread_id": st.session_state['chat_id']}}
            st.session_state.agent = initialize_agent()

            with st.spinner('Generating response...'):
                result = st.session_state.agent.invoke(
                    {"input": base_query},
                    config=config,
                )
            
            st.session_state.first_run = False
        else:
            with st.spinner('Generating response...'):
                logging.info("stored chat id " + st.session_state['chat_id'])
                config = {"configurable": {"thread_id": st.session_state['chat_id']}}
                result = st.session_state.agent.invoke(
                    {"input": prompt},
                    config=config,
                )
        
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write(result["answer"])
        
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})


if __name__ == "__main__":
    main()