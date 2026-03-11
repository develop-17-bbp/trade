import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("REASONING_LLM_KEY")

if not api_key:
    print("❌ REASONING_LLM_KEY not found in environment.")
else:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, reply with 'Gemini Online' if you can hear me.")
        print(f"✅ Response from Gemini: {response.text.strip()}")
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
