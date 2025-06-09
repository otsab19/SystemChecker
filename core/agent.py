from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from core.agent_tools import RAGQueryTool, LiveSystemInfoTool, SystemActionTool, ExternalSearchTool
from core.vector_store import VectorStoreManager
from config.settings import settings


class SystemAdminAgent:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.TEMPERATURE
        )

        # Initialize tools
        self.tools = [
            RAGQueryTool(vector_store),
            LiveSystemInfoTool(),
            SystemActionTool(),
            ExternalSearchTool()
        ]

        # Create agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    def _create_agent(self):
        """Create the ReAct agent"""
        prompt_template = """
You are an AI System Administrator Assistant. Your role is to help users with system administration tasks, troubleshooting, and optimization.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Key Guidelines:
1. Always start by checking the local system information (RAG) before getting live data
2. For system modifications, always use the system_action tool which requires user confirmation
3. Provide clear, actionable advice
4. If local information is insufficient, consider external search
5. Be proactive in suggesting optimizations and preventive measures

Question: {input}
Thought: {agent_scratchpad}
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

        return create_react_agent(self.llm, self.tools, prompt)

    def query(self, user_input: str) -> str:
        """Process user query through the agent"""
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result["output"]
        except Exception as e:
            return f"Error processing query: {str(e)}"