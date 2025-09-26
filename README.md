# Welcome to the aws-quantiica-umm-model repository!

Here you will find the agent definitions, tools and schemas related to them.

## Getting started

1. Ensure you have Python 3.12+ installed. Then, after clonning the repo create your virtual environment:

``` bash
python -m venv .venv
```

You can activate it by doing the following (if you are using Windows)

``` bash
.venv/Scripts/activate
```

or this (if you are using Mac)

``` bash
$ source .venv/bin/activate
```

2. Install dependencies:

``` bash
pip install -r requirements.txt
```

3. We use **Bedrock Runtim**e to access foundational LLM models via the **langchain-aws** boto3 wrapper and **S3 Vectors**, which is managed via boto3, as our vector database. To access these services, you must configure your credentials with the _AWS CLI_ before executing the model. You must also create your _**.env**_ file to access the correct services.

The structure of the **.env** file is as follows:

``` env
S3_VECTOR_BUCKET_NAME=
UMM_S3_VECTOR_INDEX=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
```


## Execute the Conversational Agent

The `ConversationalAgent` class in the `agents.py` file encapsulates all the basic logic of an RAG agent. To execute it, import the class and the `AgentState` class from the `schemas.py` file. Then run the static `invoke()` method, which returns an AgentState instance modified by the agent. The agent response is the last message in the `messages` list attribute.

``` python
from agent import ConversationalAgent
from schemas import AgentState
from langchain_core.messages import HumanMessage, AIMessage

messages = []

messages.append(
    HumanMessage(name="User", content="Me podrías compartir las instrucciones para inscribirme?")
)

response = ConversationalAgent.invoke(
    AgentState(
        messages=messages
    )
)

for message in response.get("messages"):
    print("------------------------")
    print(f"{message.name}\n\t{message.content}")
print("------------------------")
```

