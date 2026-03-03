"""
LangChain 프롬프트 템플릿
"""
from langchain.prompts import PromptTemplate


# RAG 프롬프트

RAG_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an financial AI assistant specialized in quantitative investing.
You analyze financial news data to help investors make data-driven decisions.

### Instructions:
- Answer based only on the given context. **Do not speculate or generate information beyond the provided data.**
- If the answer is not found in the context, say **"The given news does not contain relevant information."**
- Use clear, data-driven insights that would be helpful for quantitative investing.
- Do not include opinions or subjective analysis.
- Exclude unnecessary explanations and be concise.

### Financial News Context:
{context}

### Investor's Question:
{question}
""",
)


# 요약 프롬프트

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following news article in a concise and informative manner.
- Keep the summary short and to the point.
- Extract the most important details and main points.
- Limit the response to 2-3 sentences.
- Provide only the summary, without any additional commentary.
- Do not introduce the summary with phrases such as "Here is a summary."
- Do not include phrases like "It appears that" or "It looks like."
The output should contain only the summarized content.

News Article:
{text}

Summary:
""",
)


# 키워드 추출 프롬프트

KEYWORD_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""
Extract **1 to 5 most important keywords** from the following news article.
- Respond only with the keywords, **separated by commas**.
- The keywords should be **short and relevant**.
- Do not introduce the keyword with phrases such as "Here are the keywords."

News Article:
{text}

Keywords:
""",
)
