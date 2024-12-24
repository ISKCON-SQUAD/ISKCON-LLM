FROM python:3.10

WORKDIR /app

# copies .env if it exists (weird workaround)
COPY requirements.txt .en[v] ./
COPY *.py .
COPY src src
COPY chroma_vector_db chroma_vector_db
COPY book book

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8501

# CMD ["streamlit", "run", "main.py"]
CMD ["uvicorn", "api_app:api_app", "--host", "0.0.0.0", "--port", "8501"]