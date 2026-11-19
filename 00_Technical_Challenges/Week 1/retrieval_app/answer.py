"""Stage 5: turn retrieved chunks into an answer.

YOUR CODE GOES HERE. The prompt is yours to write — it is the cheapest thing
in the pipeline to change and often the most effective.
"""

from .index import Hit, client

CHAT_MODEL = "gpt-4.1-mini"

RAG_SYSTEM_TEMPLATE = """You are a helpful personal wellness assistant that answers health and wellness questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't have information about that in my wellness knowledge base"
- Be accurate and cite the same [chunk N] label that appears before each chunk in the context after each claim.
- Keep responses succinct, focusing on the most relevant information.
- Only use the provided context. Do not use external knowledge.
- Include a gentle reminder that users should consult healthcare professionals for medical advice when appropriate
- Only provide answers when you are confident the context supports your response."""

RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}

Question: {user_query}

Please provide your answer based solely on the context above."""

def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    """Assemble the messages list sent to the chat model.

    Args:
        question: the user's question.
        hits: what search() returned. May be empty.

    Returns:
        A list of {"role": ..., "content": ...} dicts — a system message
        setting the rules, and a user message carrying the context and the
        question.

    Three things your prompt has to decide, and they are real decisions:
      1. How do you label each chunk in the context so the model can refer to
         it? (hit.chunk_id exists for this.)
      2. What should it do when the context does not contain the answer? Say
         so explicitly, or it will answer from the nearest chunk regardless.
      3. What happens when `hits` is empty? Write that branch deliberately.

    Hint: Session 2's RAG_SYSTEM_TEMPLATE is a good starting point, and a
    better one once you have read Step 3's failures and edited it.
    """
    context_blocks = []
   
    for hit in hits:
        label = f"[chunk {hit.chunk_id}]"
        labeled_text = f"{label} {hit.text}"
        context_blocks.append(labeled_text)

    if not context_blocks:
        context = "No hits were retrieved."
    else:
        context = "\n\n".join(context_blocks)

    system_message = {"role": "system", "content": RAG_SYSTEM_TEMPLATE}
    user_message = {"role": "user", "content": RAG_USER_TEMPLATE.format(context=context, context_count=len(hits), user_query=question)}

    return [system_message, user_message]


def answer(question: str, hits: list[Hit]) -> str:
    """Call the chat model and return its reply as a string.

    Hint: `client().chat.completions.create(model=CHAT_MODEL, messages=...)`,
    then dig out `.choices[0].message.content`. Look up what `temperature`
    does to your results when you run the same question twice.
    """
    messages = build_messages(question, hits)
    response = client().chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0)
    return response.choices[0].message.content
