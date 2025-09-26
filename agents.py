
from schemas import AgentState
from tools import get_information_tool
from typing import Literal, Dict
from dotenv import load_dotenv
from time import time
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, END, StateGraph
import json

load_dotenv()

llm_haiku = ChatBedrockConverse(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    temperature=0,
    max_tokens=500,
).bind_tools([get_information_tool])


CONTEXT = """La Universidad Metropolitana de Monterrey se ha posicionado en Nuevo León como una de las instituciones educativas privadas de más rápido crecimiento, con mayor fortaleza, calidad y flexibilidad en sus planes de estudio.

## Misión
Brindar una formación integral a nuestros estudiantes a través de una oferta educativa de calidad, centrada en el aprendizaje. La formación UMM incluye conocimientos relativos a cada área de estudio, desarrollo personal, habilidades, competencias profesionales y responsabilidad social, generando así, un perfil de egreso competente y competitivo.

## Visión
Ser una institución educativa con programas de calidad, innovadores y económicamente accesibles para la comunidad.

## Valores
- Justicia
- Libertad
- Honestidad
- Respeto
- Responsabilidad Social
- Calidad
- Empatía

El **Modelo Educativo UMM** está centrado en el aprendizaje del alumno. Gira en torno a cinco aspectos pedagógicos prioritarios que son plasmados en los planes de estudio a través de asignaturas específicas y, en otros casos, a través de competencias transversales a lo largo de toda la carrera del estudiante.
La cuestión académica es el eje central y a esta se incorpora de forma paulatina y constante, el desarrollo personal de alumno. En ambos casos el maestro se convierte en un facilitador y guía de esta experiencia. El propósito de ofrecer una preparación integral es la de formar egresados capacitados, productivos, emocionalmente maduros y socialmente comprometidos.
Lo anterior se logra a través de actividades extracurriculares que tienen como objetivo favorecer el fortalecimiento de la autoestima, desarrollar el interés por actividades recreativas sanas y estimular la dimensión artística de los jóvenes.
Por otra parte, el eje exterior responde al cambiante contexto social en el que la Universidad está inmersa. De esta manera el Modelo Educativo UMM contempla un proceso constante de planeación, desarrollo, implementación y evaluación de las iniciativas educativas. Este ciclo continuo permite cumplir con el reto de trasformar, cuando sea necesario, la manera en que la institución cumple con su misión educativa, de tal forma que la evaluación y la mejora continua sean parte natural de sus procesos académicos.

"""

class ConversationalAgent:
    SYSTEM_PROMPT: Dict = {
        "initial_context": None,
        "role": "Asistente Virtual",
        "name": "Leonardo AI",
        "persona": "Un asistente virtual que trabaja para la Universidad Metropolitana de Monterrey. Tu objetivo es proporcionar información general disponible sobre la universidad, responder dudas básicas de los usuarios y, de manera discreta, identificar si existe interés en que se registren como estudiantes potenciales.",
        "tools": {
            "get_information_tool": "Te permite consultar información verificada de una base de conocimientos. Se usa cuando la información disponible no es suficiente para responder la consulta. Garantiza precisión y actualidad de las respuestas."
        },
        "reasoning_instructions": [
            "Analiza la pregunta del usuario para determinar si es básica o requiere información adicional.",
            "Decide si debes responder con el contexto inicial o utilizar 'get_information_tool'.",
            "Considera de forma discreta cómo indagar sobre el interés del usuario en inscribirse, sin sonar insistente o invasivo."
        ],
        "response_guidelines": {
            "steps": [
                "Analiza la pregunta del usuario e identifica los datos esenciales y la complejidad de la consulta.",
                "Decide si puedes responder con el contexto inicial o si debes usar 'get_information_tool'.",
                "Elabora una respuesta clara, amable y profesional, incluyendo información relevante.",
                "Añade, de forma discreta y natural, una sutil invitación a conocer más sobre el proceso de inscripción o servicios estudiantiles, según corresponda."
            ],
            "format": {
                "show_reasoning": False,
                "style": "breve, directo y cordial",
                "length": "1 a 3 párrafos máximo",
                "integration": "Si se consultó información adicional, intégrala sin mencionarlo explícitamente."
            },
            "considerations": [
                "No seas insistente con la invitación a inscribirse hazlo de manera sugerente.",
                "Prioriza siempre claridad, exactitud y profesionalismo.",
                "Utiliza 'get_information_tool' cuando la duda supere el contexto disponible."
            ]
        }
    }

    tools: Dict[str, BaseTool] = {
        "get_information_tool": get_information_tool
    }

    @staticmethod
    def make_system_message(context):
        prompt_obj = ConversationalAgent.SYSTEM_PROMPT.copy()
        prompt_obj["initial_context"] = context
        content = json.dumps(prompt_obj, ensure_ascii=False, indent=4)
        return SystemMessage(content=content)

    @staticmethod
    def invoke(state: AgentState, config=None)->AgentState:
        builder = StateGraph(AgentState)

        builder.add_node("generation", ConversationalAgent.generation_node)
        builder.add_node("execute_tool", ConversationalAgent.execute_tool_node)

        builder.add_edge(START, "generation")
        builder.add_conditional_edges(
            "generation",
            ConversationalAgent.decide_edge,
            {
                "execute_tool": "execute_tool",
                "answer": END, 
            },
        )
        builder.add_edge("execute_tool", "generation")

        chat_agent = builder.compile()

        return chat_agent.invoke(state, config)

    @staticmethod
    def generation_node(state: AgentState, config=None)->AgentState:
        print("Generation node reached...")

        try:
            messages = state.get("messages", [])[-8:]

            system_message = ConversationalAgent.make_system_message(CONTEXT)

            messages.insert(0, system_message)
            t0 = time()
            # Score each doc
            llm_response = llm_haiku.invoke(
                messages
            )
            t1 = time()

            print(f"Response generated in {t1-t0} seconds!...")
            # print(f"Response:\n{llm_response}")

            llm_response.name = "Leonardo AI"
            state.get("messages", []).append(llm_response)
        except Exception as e:
            print(f"An error has ocurred in generation node: {e}")
            raise ValueError("Generation node must return a LLM message")

        return state

    @staticmethod
    def execute_tool_node(state: AgentState, config=None)->AgentState:
        """
        Executes the model decided tool

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): tool_response key is added to the state dict
        """

        last_message: AIMessage = state.get("messages")[-1]

        tool_call_tries = state.get("tool_calls_tries", 0)

        tool_call = last_message.tool_calls[0]

        tool_name = tool_call["name"]

        tool = ConversationalAgent.tools.get(tool_name)

        tool_message: ToolMessage = tool.invoke(tool_call)

        # print(tool_message)
        
        state["tool_calls_tries"] = tool_call_tries + 1

        artifact = tool_message.artifact
        if artifact:
            state.update(**artifact)

        state["messages"].append(tool_message)
        
        return state

    @staticmethod
    def decide_edge(state: AgentState, config=None)->Literal["answer", "execute_tool"]:
        """
        ## Grade Documents

        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        tool_call_tries = state.get("tool_calls_tries", 0)

        last_message = state.get("messages", [])[-1]

        if not last_message or last_message.type != "ai":
            raise ValueError("Last message must be AI type")

        if not last_message.tool_calls or tool_call_tries > 0:
            return "answer"
        
        return "execute_tool"

    