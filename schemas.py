from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import TypedDict, List, Literal

class GetInformationInput(BaseModel):
    """Input for get information queries."""
    query: str = Field(description="Information you want to obtain from the knowledge base")
    modality: Literal["AulaFlex", "Ejecutivo AulaFlex", "En Línea", "Fin de Semana", ""] = Field(
        description="""Program's Modality you want to view information about (if applicable). 
If you are not looking for information about a program, you can select ''."""
    )
    program_type: Literal["profesional", "prepa-umm", "posgrado", ''] = Field(
        description="Level of studies of the program you want to consult information about (if applicable). If you are not looking for information about a program, you can select ''"
    )

class AgentState(TypedDict):
    messages: List[BaseMessage]
    tool_calls_tries: int = 0
    tool_response: int


