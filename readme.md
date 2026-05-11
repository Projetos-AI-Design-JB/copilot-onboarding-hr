
# Enterprise Copilot: Onboarding de RH / HR Onboarding (RAG Architecture)

*(English version below)*

## 🇧🇷 [PT-BR] Visão Executiva
Este projeto é um assistente virtual corporativo projetado para resolver a ineficiência no processo de integração de novos colaboradores. Utilizando a arquitetura RAG (Retrieval-Augmented Generation), o sistema responde a dúvidas de RH consultando exclusivamente os documentos oficiais da empresa. O foco central do produto é a governança de IA, garantindo previsibilidade, segurança de dados e eficiência de custos.

## Metodologia
A construção seguiu o framework Spec-Driven Development (SDD). A arquitetura técnica e os requisitos de negócio foram estritamente definidos através de um PRD (Product Requirement Document) e de Jornadas Críticas do Usuário (CUJs) antes da geração do código, garantindo alinhamento total entre a necessidade do mercado e a execução da engenharia.

## Stack Tecnológico
* **API e Backend:** FastAPI (Python) operando como um microserviço isolado.
* **Inteligência Artificial:** Google Cloud Vertex AI (modelos `gemini-1.5-flash` para inferência e `text-embedding-005` para vetorização).
* **Memória Semântica:** Pinecone (Banco Vetorial) para armazenamento e busca rápida de contexto.
* **Interface de Usuário:** Streamlit para validação de negócio e demonstração interativa.

## Pilares de Governança (AI Compliance)
* **Zero Alucinação:** A temperatura do modelo generativo está fixada em 0.1 e o prompt possui regras estritas de Role Prompting, garantindo que a IA não invente políticas ou benefícios inexistentes.
* **Segurança e Controle:** Implementação de Middleware de autorização exigindo tokens de API válidos em todas as requisições, prevenindo acessos não autorizados a dados corporativos.
* **Observabilidade:** Logs estruturados em todas as camadas da aplicação (API, vetorização, recuperação e geração) para permitir auditoria, rastreabilidade e monitoramento de performance.
* **Eficiência Operacional:** Validação de payload no backend e preparação para deploy em ambiente Serverless (Google Cloud Run), garantindo escala e baixo custo computacional.

## Estrutura do Projeto
* `ingest_knowledge.py`: Pipeline de extração, vetorização e ingestão de dados corporativos no Pinecone.
* `rag_engine.py`: Motor central de inteligência e recuperação de contexto.
* `api.py`: Microserviço backend com rotas protegidas.
* `app.py`: Interface de usuário em Streamlit.

## Como Executar Localmente

**1. Instalação de Dependências**
```bash
pip install -r requirements.txt
```

**2. Variáveis de Ambiente**
Configure as seguintes credenciais no seu terminal (ou arquivo `.env`):
* `GOOGLE_CLOUD_PROJECT`: ID do seu projeto no GCP.
* `PINECONE_API_KEY`: Sua chave de acesso do Pinecone.
* `API_TOKEN`: O token de segurança para acessar o backend (ex: "seu-token-seguro").

**3. Iniciar os Serviços**
Abra dois terminais distintos.

No Terminal 1, inicie a API:
```bash
uvicorn api:app --reload
```

No Terminal 2, inicie a Interface de Usuário:
```bash
streamlit run app.py
```

---

## 🇺🇸 [EN] Executive Vision
This project is a corporate virtual assistant designed to solve inefficiencies in the new employee onboarding process. Using the RAG (Retrieval-Augmented Generation) architecture, the system answers HR questions by exclusively consulting the company's official documents. The core focus of the product is AI governance, ensuring predictability, data security, and cost efficiency.

## Methodology
The build followed the Spec-Driven Development (SDD) framework. The technical architecture and business requirements were strictly defined through a PRD (Product Requirement Document) and Critical User Journeys (CUJs) before any code generation, ensuring total alignment between market needs and engineering execution.

## Tech Stack
* **API and Backend:** FastAPI (Python) operating as an isolated microservice.
* **Artificial Intelligence:** Google Cloud Vertex AI (models `gemini-1.5-flash` for inference and `text-embedding-005` for vectorization).
* **Semantic Memory:** Pinecone (Vector Database) for storage and rapid context retrieval.
* **User Interface:** Streamlit for business validation and interactive demonstration.

## Governance Pillars (AI Compliance)
* **Zero Hallucination:** The generative model's temperature is set to 0.1 and the prompt has strict Role Prompting rules, ensuring the AI does not invent non-existent policies or benefits.
* **Security and Control:** Implementation of an authorization Middleware requiring valid API tokens for all requests, preventing unauthorized access to corporate data.
* **Observability:** Structured logs across all application layers (API, vectorization, retrieval, and generation) to allow auditing, traceability, and performance monitoring.
* **Operational Efficiency:** Payload validation on the backend and preparation for Serverless deployment (Google Cloud Run), ensuring scale and low computational cost.

## Project Structure
* `ingest_knowledge.py`: Pipeline for extraction, vectorization, and ingestion of corporate data into Pinecone.
* `rag_engine.py`: Core intelligence and context retrieval engine.
* `api.py`: Backend microservice with protected routes.
* `app.py`: User interface built with Streamlit.

## How to Run Locally

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Environment Variables**
Configure the following credentials in your terminal (or `.env` file):
* `GOOGLE_CLOUD_PROJECT`: Your GCP project ID.
* `PINECONE_API_KEY`: Your Pinecone access key.
* `API_TOKEN`: The security token to access the backend (e.g., "your-secure-token").

**3. Start the Services**
Open two separate terminals.

In Terminal 1, start the API:
```bash
uvicorn api:app --reload
```

In Terminal 2, start the UI:
```bash
streamlit run app.py
```