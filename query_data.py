import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
load_dotenv()
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are DesignThinkingBuddy, a design thinking expert and an assistant to anybody that is trying to generate and work on a new idea. You must be spot on, concise and to the point.
You are also a expert at project initiation, brainstorming, and idea generation. You are here to help users with design thinking tools and techniques. While providing guidance, 
you should also encourage users to explore creative possibilities and think outside the box. You should also provide links to tools when they are available.

You should refer to casestudes in the data that I have shared. 

Now, using only the following context to help the user:
{context}

Previous conversation:
{history}

Question: {question}

As DesignThinkingBuddy, a helpful guide, provide a relevant, and accurate response to help someone who is starting a business. 
Make sure you are spot on and provide a focused question to continue the conversation, acknowledging the previous turn. Include links to tools when available, and only use the provided context.

The language should be simple and easy to understand, and the response should be concise and to the point. 

I need you to provide the response in a way where you use bullet point if you need to but make it easy to understand and not just a huge paragraph, put the bullet point in the next line so its easier to understand
Give spaces between the bullet points so its easier to read and understand.
"""

IMAGE_TEMPLATE = """
You are DesignThinkingBuddy, an expert at analyzing various design thinking tools and visual data. You must be concise, spot-on and to the point. 

Analyze the following image content:
{user_input}

First, determine if this image represents a tool from innovationhubslc.ca/tools. If so, please identify which tool it is.
 
Based on the above determination, provide the following analysis:
    If it's a known tool:
     - A detailed analysis of the data provided for this tool.
        - Specifically analyze the step sequence validity, action consistency, location/state conflicts, and specific improvement needs.
     If it's not a known tool:
       - Provide feedback relevant to the context provided in my training data.
"""

def process_query(query_text, chat_history=None):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
   
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if not results:
        return {"error": "No relevant information found"}, chat_history
       
    context = "\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    if chat_history:
        formatted_history = "\n".join([f"User: {q}\nBot: {r}" for q, r in chat_history[-3:]])
    else:
        formatted_history = ""
        
    response = ChatOpenAI().predict(prompt.format(
       context=context,
       question=query_text,
       history = formatted_history
    ))

    updated_history = chat_history if chat_history else []
    updated_history.append((query_text, response))
   
    return {"response": response}, updated_history

def analyze_journey_content(sections):
   steps = []
   issues = []
   current_state = {'location': 'outside', 'transport': None}
   
   for i in range(1, 6):
       step_key = f'STEP {i}'
       if step_key in sections:
           step = sections[step_key].lower()
           steps.append(step)
           
           if 'bus' in step:
               if 'boards' in step:
                   if current_state['transport'] == 'bus':
                       issues.append(f"Step {i}: Already on bus")
                   current_state['transport'] = 'bus'
                   current_state['location'] = 'inside'
               
           if current_state['transport'] == 'bus':
               if 'uber' in step or 'taxi' in step:
                   issues.append(f"Step {i}: Cannot take Uber/taxi while on bus")
               if 'waits' in step:
                   issues.append(f"Step {i}: Cannot wait for bus while on bus")
                   
           if 'seat' in step and current_state['location'] != 'inside':
               issues.append(f"Step {i}: Cannot find seat before boarding")

   return steps, issues

def extract_and_analyze_content(image_text):
   sections = {}
   current_section = None
   content_lines = []

   for line in image_text.split('\n'):
       line = line.strip()
       if not line: continue

       if 'STEP' in line:
           if current_section:
               sections[current_section] = '\n'.join(content_lines)
           current_section = line
           content_lines = []
       elif current_section:
           content_lines.append(line)

   if current_section:
       sections[current_section] = '\n'.join(content_lines)

   steps, issues = analyze_journey_content(sections)
   return sections, issues

def process_image(image_path):
   try:
       image = Image.open(image_path)
       image_text = pytesseract.image_to_string(image)
       
       sections, issues = extract_and_analyze_content(image_text)
       
       prompt = ChatPromptTemplate.from_template(IMAGE_TEMPLATE)
       response = ChatOpenAI().predict(prompt.format(user_input=str(sections)))
       
       return {
           "sections": sections,
           "issues": issues, 
           "response": response
       }
       
   except Exception as e:
       return {"error": f"Error: {str(e)}"}

def chat():
    chat_history = []
    print("Hi there! 🌟 I'm DesignThinkingBuddy. Let's turn your ideas into action! What can I do for you?")
    
    while True:
        query_text = input("You: ")
        if query_text.lower() == "exit":
            break

        result, chat_history = process_query(query_text, chat_history)

        if "error" in result:
           print(f"Bot: Error: {result['error']}")
        else:
           print("Bot:", result["response"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--query", type=str, help="Text query")
    args = parser.parse_args()

    if args.image:
        result = process_image(args.image)
        print("\n" + "="*50)
        if "error" in result:
           print(f"\nError: {result['error']}")
        else:
            if "sections" in result:
               print("\nContent:")
               for section, content in result["sections"].items():
                   print(f"\n{section}: {content}")
            if "issues" in result:
                print("\nIssues:")
                for issue in result["issues"]:
                    print(f"- {issue}")
            print("\nAnalysis:")
            print(result["response"])
        print("\n" + "="*50)
    elif args.query:
        result, _ = process_query(args.query)
        print("\n" + "="*50)
        if "error" in result:
            print(f"\nError: {result['error']}")
        else:
            print("\nAnalysis:")
            print(result["response"])
        print("\n" + "="*50)
    else:
        chat()

if __name__ == "__main__":
    main()