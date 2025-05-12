import json
import importlib
from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, Optional
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.tool import ToolMessage
import logging
from pathlib import Path
from typing import Union
from agentools.register.tool import ToolManager, register_function

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentMeta(ABC):
    """Abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List = [],
        *args,
        **kwargs,
    ):
        """Initialize a new Agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """Synchronously invoke the agent's main function"""
        pass

    @abstractmethod
    async def invoke_async(self, query: str, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's main function"""
        pass


class Agent(AgentMeta):
    """Concrete implementation of an AI agent with tool-calling capabilities"""

    TOOLS_PATH = Path("tool_template/tools.json")

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List = [],
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        skills: list[str] = ["You can answer the user question with tools"],
        *args,
        **kwargs,
    ):
        """
        Initialize the agent with a language model, a list of tools, a description, and a set of skills.
        Parameters:
        ----------
        llm : Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI]
            An instance of a language model used by the agent to process and generate responses.

        tools : List, optional
            A list of tools that the agent can utilize when performing tasks. Defaults to an empty list.

        description : str, optional
            A brief description of the assistant's capabilities. Defaults to a general helpful assistant message.

        skills : list[str], optional
            A list of skills or abilities describing what the assistant can do. Defaults to a basic tool-usage skill.

        *args, **kwargs : Any
            Additional arguments passed to the superclass or future extensions.
        """

        self.llm = llm
        self.tools = tools
        self.description = description
        self.skills = skills
        self.TOOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.register_tools(self.tools)

    def register_tools(self, tools: List[str]) -> Any:
        """
        Register a list of tools
        """
        exisiting_tools = ToolManager.load_tools()
        list_modules = set([info["module_path"] for info in exisiting_tools.values()])
        for tool in tools:
            if tool not in list_modules:
                register_function(tool)
            else:
                logger.warning(f"Tool {tool} already registered. Skipping registration.")

    def invoke(self, query: str, *args, **kwargs) -> Any:
        """
        Select and execute a tool based on the task description
        """
        try:
            tools = json.loads(self.TOOLS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tools = {}
            self.TOOLS_PATH.write_text(json.dumps({}, indent=4), encoding="utf-8")

        prompt = (
            "You are given a task and a list of available tools.\n"
            f"- Task: {query}\n"
            f"- Tools list: {json.dumps(tools)}\n\n"
            "Instructions:\n"
            "- If the task can be solved without tools, just return the answer without any explanation\n"
            "- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n"
            "{\n"
            '"tool_name": "Function name",\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool_name",\n'
            '"module_path": "Path to import the tool"\n'
            "}\n"
            "Let's say I don't know and suggest where to search if you are unsure the answer.\n"
            "Not make up anything.\n"
        )
        skills = "- ".join(self.skills)
        messages = [
            SystemMessage(content=f"{self.description}\nHere is your skills: {skills}"),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            tool_data = self._extract_json(response.content)

            if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                return response

            tool_call = json.loads(tool_data)
            return self._execute_tool(
                tool_call["tool_name"], tool_call["arguments"], tool_call["module_path"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None

    async def invoke_async(self, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's LLM"""
        return await self.llm.ainvoke(*args, **kwargs)

    def _execute_tool(self, tool_name: str, arguments: dict, module_path: str) -> Any:
        """Execute the specified tool with given arguments"""
        # If function is directly registered by decorator @function_tool. Access it on runtime context.
        registered_functions = ToolManager.load_tools()

        if (
            module_path == "__runtime__"
            and tool_name in ToolManager._registered_functions
        ):
            func = ToolManager._registered_functions[tool_name]
            content = f"Completed executing tool {tool_name}({arguments})"
            logger.info(content)
            artifact = func(**arguments)
            tool_call_id = registered_functions[tool_name]["tool_call_id"]
            message = ToolMessage(
                content=content, artifact=artifact, tool_call_id=tool_call_id
            )
            return message

        # If function is imported from a module, access it on module path.
        # try:
        if tool_name in globals():
            return globals()[tool_name](**arguments)

        module = importlib.import_module(module_path, package=__package__)
        func = getattr(module, tool_name)
        content = f"Completed executing tool {tool_name}({arguments})"
        logger.info(content)
        artifact = func(**arguments)
        tool_call_id = registered_functions[tool_name]["tool_call_id"]
        message = ToolMessage(
            content=content, artifact=artifact, tool_call_id=tool_call_id
        )
        return message
        # except (ImportError, AttributeError) as e:
        #     logger.error(f"Error executing tool {tool_name}: {str(e)}")
        #     return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract first valid JSON object from text using stack-based parsing"""
        start = text.find("{")
        if start == -1:
            return None

        stack = []
        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None
