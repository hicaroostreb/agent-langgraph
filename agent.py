from pydantic import BaseModel, Field
from typing import Optional, List
from trustcall import create_extractor
from database.pg_vector import SupabaseVectorDB

from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

import configuration

# --------------------- LLM SETUP ---------------------
model = ChatVertexAI(model="gemini-2.0-flash-lite-001", temperature=0, max_tokens=200)

# --------------------- EMBEDDING SETUP ---------------------
hf = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# --------------------- USER PROFILE ---------------------
class UserProfile(BaseModel):
    nome: Optional[str] = Field(None)
    sobrenome: Optional[str] = Field(None)
    email: Optional[str] = Field(None)
    telefone: Optional[str] = Field(None)
    necessidade: Optional[str] = Field(None)
    valor_desejado: Optional[str] = Field(None)
    urgencia: Optional[str] = Field(None)
    nivel_conhecimento_consorcio: Optional[str] = Field(None)
    disponibilidade_lance: Optional[str] = Field(None)
    finalidade: Optional[str] = Field(None)
    orcamento_mensal: Optional[str] = Field(None)
    tomada_decisao: Optional[str] = Field(None)


# --------------------- TRUSTCALL EXTRACTOR ---------------------
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile",
)

# --------------------- PROMPT SETUP ---------------------
# Agent instruction
MODEL_SYSTEM_MESSAGE = """
Você é um agente de atendimento e qualificação (SDR) especializado em consórcios. Seu papel é acolher leads de forma humanizada e conduzir uma conversa leve, 
estratégica e consultiva, com foco em entender o momento do cliente e se ele está apto para avançar para um especialista.

Diretrizes:
1. NUNCA use emojis, linguagem informal ou gírias.
2. Nunca comece falando sobre consórcio. Dê espaço para o usuário guiar a conversa no início.
3. Não faça perguntas diretas, mas sim perguntas abertas que incentivem o lead a compartilhar informações.
4. Nunca faça mais de uma pergunta junto. Pergunte uma coisa de cada vez.
5. Sempre consulte a memória antes de fazer uma pergunta.
6. Se uma informação já estiver na memória, utilize-a para avançar ou validar, e nunca repita a pergunta como se fosse nova.
7. Evite soar robótico. Use linguagem natural e consultiva ao usar ou validar dados já informados.

Se houver informações salvas na memória sobre este usuário, siga estas orientações:

• Nunca peça diretamente algo que já está salvo.
• Utilize as informações para personalizar a conversa, demonstrando atenção e continuidade.
• Quando necessário, valide sutilmente, por exemplo: “Pelo que entendi, sua ideia é...” ou “Se for isso mesmo...”.
• Consulte a memória antes de cada nova pergunta para evitar repetições e manter o fluxo natural.

Aqui está a memória (talvez esteja vazia): {memory}

Seu papel é entender o momento do lead, educar sobre consórcio quando necessário e extrair, ao longo da conversa, as informações da memória.

INFORMAÇÕES TÉCNICAS RELEVANTES:
{rag_context}

Use as informações técnicas acima quando forem relevantes para responder às perguntas do usuário sobre consórcios.
Se as informações técnicas não forem relevantes para a pergunta atual, ignore-as e responda naturalmente.

Foque em coletar os seguintes dados, naturalmente ao longo da conversa:
• [Necessidade principal]
• [Valor desejado do bem ou negócio]
• [Urgência do plano]
• [Nível de conhecimento sobre consórcios]
• [Disponibilidade de dar lance]
• [Finalidade: uso próprio ou investimento]
• [Orçamento mensal disponível]
• [Forma de tomada de decisão: forma de decidir, o que leva em consideração, como funciona o processo decisório]
"""

# Extraction instruction
TRUSTCALL_INSTRUCTION = """
Você é um agente responsável por atualizar a memória (JSON doc) do usuário com base na conversa abaixo.

Sua tarefa é revisar a conversa e preencher ou atualizar os seguintes campos no perfil do usuário:

- nome: Primeiro nome do usuário. Ex: "Carlos"
- sobrenome: Último sobrenome do usuário. Ex: "Silva"
- email: Endereço de e-mail fornecido. Ex: "carlos@email.com"
- telefone: Telefone informado pelo usuário. Ex: "11987654321"
- necessidade: O que o usuário deseja adquirir com o consórcio. Ex: "Comprar uma casa", "Comprar um carro", "Investir"
- valor_desejado: Valor aproximado do bem ou objetivo. Ex: "500k", "R$ 80.000", "300 mil"
- urgencia: Prazo ou horizonte desejado para alcançar o objetivo. Ex: "5 anos", "1 ano", "curto prazo"
- nivel_conhecimento_consorcio: Grau de familiaridade do usuário com consórcios. Ex: "iniciante", "intermediário", "experiente"
- disponibilidade_lance: Se o usuário tem ou pretende oferecer lance. Ex: "Não possui", "Tem reserva", "Pretende dar lance"
- finalidade: Se o objetivo é para uso pessoal ou investimento. Ex: "Uso próprio", "Investimento", "Aquisição de imóvel para moradia"
- orcamento_mensal: Valor mensal que o usuário pode pagar. Ex: "1000", "até 300 reais", "R$ 500"
- tomada_decisao: Como o usuário costuma tomar decisões. Ex: "Sozinho", "Em conjunto com a esposa", "Com o sócio", "Indefinido"

Instruções importantes:

1. Se algum campo não foi mencionado diretamente pelo usuário, deixe-o como "Desconhecido".
2. Se alguma informação puder ser inferida com clareza e segurança, registre-a. Exemplos:
   - "geralmente decido junto com minha esposa" → tomada_decisao: "Em conjunto com a esposa"
   - "sem pressa" → urgencia: "baixa"
3. Caso não haja segurança na inferência, mantenha o campo como "Desconhecido".
4. Dê preferência por termos objetivos, curtos e descritivos.
5. Certifique-se de preencher o máximo possível com base nas mensagens.

Retorne apenas o objeto JSON `UserProfile` com os campos preenchidos ou atualizados.
"""

# --------------------- VETOR DB INSTANCE ---------------------
vector_db = SupabaseVectorDB()


# --------------------- RAG RETRIEVAL ---------------------
def get_rag_retrieval(query: str) -> str:
    try:
        # Etapa 1: Geração do embedding
        processed_query = query.strip().lower()
        query_embedding = hf.embed_query(processed_query)

        # Etapa 2: Busca vetorial
        results = vector_db.search_similar_faqs(
            query_embedding=query_embedding, top_k=3
        )

        if not results:
            return "Nenhuma informação relevante encontrada."

        response = [f"Q: {r['pergunta']}\nA: {r['resposta']}" for r in results]
        return "\n\n---\n\n".join(response)

    except Exception as e:
        return f"Erro ao buscar informações de suporte técnico: {e}"


# --------------------- CHATBOT NODE ---------------------
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    if existing_memory and existing_memory.value:
        mem = existing_memory.value
        formatted_memory = (
            f"Nome: {mem.get('nome', 'Desconhecido')}\n"
            f"Sobrenome: {mem.get('sobrenome', 'Desconhecido')}\n"
            f"E-mail: {mem.get('email', 'Desconhecido')}\n"
            f"Telefone: {mem.get('telefone', 'Desconhecido')}\n"
            f"Necessidade: {mem.get('necessidade', 'Desconhecida')}\n"
            f"Valor Desejado: {mem.get('valor_desejado', 'Desconhecido')}\n"
            f"Urgência: {mem.get('urgencia', 'Desconhecida')}\n"
            f"Nível de Conhecimento sobre Consórcio: {mem.get('nivel_conhecimento_consorcio', 'Desconhecido')}\n"
            f"Disponibilidade de Lance: {mem.get('disponibilidade_lance', 'Desconhecida')}\n"
            f"Finalidade: {mem.get('finalidade', 'Desconhecida')}\n"
            f"Orçamento Mensal: {mem.get('orcamento_mensal', 'Desconhecido')}\n"
            f"Tomada de Decisão: {mem.get('tomada_decisao', 'Desconhecida')}"
        )
    else:
        formatted_memory = "Nenhuma informação disponível ainda."

    user_message = state["messages"][-1].content
    print(
        f"Mensagem do usuário: {user_message}"
    )  # Debug: Verificar a mensagem do usuário
    rag_context = get_rag_retrieval(user_message)

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        memory=formatted_memory, rag_context=rag_context
    )
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])

    print(f"Resposta do modelo: {response}")  # Debug: Verificar a resposta do modelo
    return {"messages": response}


# --------------------- MEMÓRIA NODE ---------------------
def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace = ("memory", user_id)

    # Carregar memória existente
    existing_memory = store.get(namespace, "user_memory")
    existing_profile = (
        {"UserProfile": existing_memory.value} if existing_memory else None
    )

    # Log para verificar o perfil existente
    print(f"Perfil existente: {existing_profile}")

    # Chamada ao extrator de dados para atualizar a memória
    result = trustcall_extractor.invoke(
        {
            "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]
            + state["messages"],
            "existing": existing_profile,
        }
    )

    # Log para inspeção do resultado da extração
    print(f"Resultado da extração: {result}")
    print(
        f"Respostas extraídas: {result.get('responses', [])}"
    )  # Verificar se a lista está vazia

    # Log para verificar as mensagens do usuário antes de invocar o modelo
    print(f"Mensagens do usuário: {state['messages']}")

    # Verificação antes de acessar o índice 0 da lista de respostas
    if result.get("responses"):
        updated_profile = result["responses"][0].model_dump()
        store.put(namespace, "user_memory", updated_profile)
    else:
        print("Nenhuma resposta válida encontrada para atualizar a memória.")


# --------------------- GRAPH ---------------------
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()
