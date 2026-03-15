
FROM python:3.13-slim

WORKDIR /frontend

# Install uv for dependency management
RUN pip install uv

COPY . /frontend

# Install dependencies via uv (system-wide)
RUN uv pip install --system -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
