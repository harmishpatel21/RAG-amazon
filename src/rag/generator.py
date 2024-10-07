import openai

def generate_response(query, context):
    prompt = f"Given the following product information:\n\n{context}\n\nAnswer the following question: {query}"
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].text.strip()