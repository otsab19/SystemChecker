from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from langchain.agents import (
    AgentExecutor, create_react_agent, create_structured_chat_agent,
    create_self_ask_with_search_agent, create_openai_functions_agent
)
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage
from langchain.tools import BaseTool

from config.settings import settings, AgentPattern, AgentMode
from core.agent_tools import get_all_tools
from core.vector_store import VectorStoreManager
from core.memory_manager import MemoryManager
from core.cache_manager import CacheManager


class BaseAgentPattern(ABC):
    """Abstract base class for agent patterns"""

    def __init__(self, vector_store: VectorStoreManager, tools: List[BaseTool]):
        self.vector_store = vector_store
        self.tools = tools
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager() if settings.ENABLE_CACHING else None

    @abstractmethod
    def create_agent(self) -> AgentExecutor:
        """Create and return the agent executor"""
        pass

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        """Get the prompt template for this pattern"""
        pass

    def execute_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a query with caching and error handling"""
        if self.cache_manager:
            cached_result = self.cache_manager.get(query)
            if cached_result:
                return cached_result

        try:
            agent = self.create_agent()
            result = agent.invoke({
                "input": query,
                **(context or {})
            })

            if self.cache_manager:
                self.cache_manager.set(query, result)

            return result
        except Exception as e:
            return {
                "output": f"Error executing query: {str(e)}",
                "error": True,
                "exception": str(e)
            }


class ReactAgentPattern(BaseAgentPattern):
    """ReAct (Reasoning + Acting) Pattern Implementation"""

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
You are an AI System Administrator Assistant using the ReAct (Reasoning + Acting) approach.

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

Guidelines:
1. Always think step by step before taking action
2. Use RAG query first to check existing system knowledge
3. Get live data when current information is needed
4. For system modifications, always use system_action tool with user confirmation
5. Provide clear, actionable advice with reasoning
6. If uncertain, ask clarifying questions or search externally

Question: {input}
Thought: {agent_scratchpad}
""",
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

    def create_agent(self) -> AgentExecutor:
        prompt = self.get_prompt_template()
        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=settings.MAX_ITERATIONS,
            handle_parsing_errors=True,
            early_stopping_method="generate"
        )


class PlanExecuteAgentPattern(BaseAgentPattern):
    """Plan-and-Execute Pattern Implementation using ReAct with planning prompts"""

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
You are an AI System Administrator that uses a Plan-and-Execute approach.

PHASE 1 - PLANNING:
First, analyze the user's request and create a detailed step-by-step plan.
Break down complex tasks into smaller, manageable steps.
Identify what tools and information you'll need for each step.

PHASE 2 - EXECUTION:
Execute each step of your plan systematically.
Use the available tools to gather information and perform actions.
Adapt your plan if you discover new information during execution.

Available tools:
{tools}

Use this format:
Question: {input}
Thought: Let me create a plan to solve this systematically
Plan:
1. [First step with reasoning]
2. [Second step with reasoning]
...

Now I'll execute this plan step by step.

Thought: I'll start with step 1 of my plan
Action: [tool to use]
Action Input: [input for the tool]
Observation: [result]
Thought: [analysis of result and next step]
Action: [next tool]
Action Input: [input]
Observation: [result]
...
Thought: I have completed my plan and can now provide the final answer
Final Answer: [comprehensive answer based on plan execution]

Question: {input}
{agent_scratchpad}
""",
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

    def create_agent(self) -> AgentExecutor:
        prompt = self.get_prompt_template()
        agent = create_react_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=settings.MAX_ITERATIONS * 2,  # More iterations for planning
            handle_parsing_errors=True,
            early_stopping_method="generate"
        )


class MultiAgentPattern(BaseAgentPattern):
    """Multi-Agent Collaborative Pattern"""

    def __init__(self, vector_store: VectorStoreManager, tools: List[BaseTool]):
        super().__init__(vector_store, tools)
        self.specialist_agents = self._create_specialist_agents()
        self.coordinator_agent = self._create_coordinator_agent()

    def _create_specialist_agents(self) -> Dict[str, AgentExecutor]:
        """Create specialized agents for different domains"""
        specialists = {}

        # Performance Monitoring Specialist
        performance_tools = [tool for tool in self.tools if 'live_system' in tool.name or 'rag_query' in tool.name]
        specialists['performance'] = self._create_specialist_agent(
            "Performance Monitoring Specialist",
            "Expert in system performance analysis, resource monitoring, and optimization",
            performance_tools
        )

        # Security Specialist
        security_tools = [tool for tool in self.tools if 'system_action' in tool.name or 'external_search' in tool.name]
        specialists['security'] = self._create_specialist_agent(
            "Security Specialist",
            "Expert in system security, vulnerability assessment, and security best practices",
            security_tools
        )

        # Troubleshooting Specialist
        troubleshooting_tools = self.tools  # Has access to all tools
        specialists['troubleshooting'] = self._create_specialist_agent(
            "Troubleshooting Specialist",
            "Expert in diagnosing and resolving system issues, log analysis, and problem solving",
            troubleshooting_tools
        )

        return specialists

    def _create_specialist_agent(self, name: str, description: str, tools: List[BaseTool]) -> AgentExecutor:
        """Create a specialized agent"""
        prompt = PromptTemplate(
            template=f"""
You are a {name}.
{description}

You work as part of a team of AI specialists. Your role is to provide expert analysis and recommendations in your domain.

Available tools:
{{tools}}

Use the ReAct format:
Question: {{input}}
Thought: [your analysis from specialist perspective]
Action: [tool to use if needed]
Action Input: [input for tool]
Observation: [result]
Thought: [analysis of result]
Final Answer: [your specialist recommendation]

Focus on your expertise area and provide detailed technical analysis.

Question: {{input}}
{{agent_scratchpad}}
""",
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
            }
        )

        agent = create_react_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

    def _create_coordinator_agent(self) -> AgentExecutor:
        """Create coordinator agent that manages specialists"""
        prompt = PromptTemplate(
            template="""
You are the Coordinator Agent for a team of AI System Administration specialists.

Your team consists of:
- Performance Monitoring Specialist: Handles performance analysis and optimization
- Security Specialist: Handles security analysis and recommendations  
- Troubleshooting Specialist: Handles problem diagnosis and resolution

Your role:
1. Analyze the user's request: {input}
2. Review specialist responses: {specialist_responses}
3. Synthesize their expertise into a comprehensive answer
4. Resolve any conflicts between specialist recommendations
5. Provide a unified, actionable response

Provide a well-structured final answer that combines the best insights from all specialists.

Final Coordinated Response:
""",
            input_variables=["input", "specialist_responses"]
        )

        # Coordinator doesn't need tools, just synthesizes responses
        agent = create_react_agent(self.llm, [], prompt)
        return AgentExecutor(agent=agent, tools=[], verbose=True, max_iterations=3)

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="Multi-agent coordination prompt",
            input_variables=["input"]
        )

    def create_agent(self) -> AgentExecutor:
        # This is handled by execute_query method for multi-agent
        return self.coordinator_agent

    def execute_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute query using multi-agent collaboration"""
        try:
            # Determine which specialists to consult based on query
            relevant_specialists = self._determine_relevant_specialists(query)

            # Get responses from relevant specialists
            specialist_responses = {}
            for specialist_name in relevant_specialists:
                if specialist_name in self.specialist_agents:
                    response = self.specialist_agents[specialist_name].invoke({"input": query})
                    specialist_responses[specialist_name] = response["output"]

            # Coordinate responses
            coordination_input = {
                "input": query,
                "specialist_responses": "\n\n".join([
                    f"{name.title()} Specialist Response:\n{response}"
                    for name, response in specialist_responses.items()
                ])
            }

            final_response = self.coordinator_agent.invoke(coordination_input)

            return {
                "output": final_response["output"],
                "specialist_responses": specialist_responses,
                "coordination_used": True
            }

        except Exception as e:
            return {
                "output": f"Error in multi-agent execution: {str(e)}",
                "error": True
            }

    def _determine_relevant_specialists(self, query: str) -> List[str]:
        """Determine which specialists are relevant for the query"""
        query_lower = query.lower()
        specialists = []

        # Performance-related keywords
        if any(keyword in query_lower for keyword in
               ['cpu', 'memory', 'disk', 'performance', 'slow', 'usage', 'resource']):
            specialists.append('performance')

        # Security-related keywords
        if any(keyword in query_lower for keyword in
               ['security', 'vulnerability', 'firewall', 'malware', 'attack', 'breach']):
            specialists.append('security')

        # Troubleshooting-related keywords
        if any(keyword in query_lower for keyword in
               ['error', 'problem', 'issue', 'fix', 'troubleshoot', 'debug', 'crash']):
            specialists.append('troubleshooting')

        # If no specific domain detected, use all specialists
        if not specialists:
            specialists = ['performance', 'security', 'troubleshooting']

        return specialists


class ConversationalAgentPattern(BaseAgentPattern):
    """Conversational Agent with Memory"""

    def __init__(self, vector_store: VectorStoreManager, tools: List[BaseTool]):
        super().__init__(vector_store, tools)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )

    def get_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an AI System Administrator Assistant with conversational capabilities.

You maintain context from previous conversations and can reference past interactions.
You have access to these tools: {tools}

Guidelines:
1. Remember previous conversations and build upon them
2. Ask follow-up questions when appropriate
3. Provide personalized recommendations based on conversation history
4. Be helpful, friendly, and professional

Use the ReAct format when using tools:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]
Observation: [tool result]
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

    def create_agent(self) -> AgentExecutor:
        prompt = self.get_prompt_template()

        # Format tools for the prompt
        formatted_prompt = prompt.partial(
            tools="\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        )

        agent = create_structured_chat_agent(self.llm, self.tools, formatted_prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=settings.MAX_ITERATIONS,
            handle_parsing_errors=True
        )


class SelfAskAgentPattern(BaseAgentPattern):
    """Self-Ask with Search Pattern"""

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
You are an AI System Administrator that uses self-questioning to solve problems.

When faced with a complex question, break it down by asking yourself sub-questions.

Available tools: {tools}

Use this format:
Question: {input}
Are follow up questions needed here: Yes/No
Follow up: [sub-question if needed]
Intermediate answer: [answer to sub-question using tools if needed]
... (repeat as needed)
So the final answer is: [final comprehensive answer]

If you need to use tools, use the ReAct format:
Thought: [reasoning]
Action: [tool name]
Action Input: [input]
Observation: [result]

Question: {input}
{agent_scratchpad}
""",
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            }
        )

    def create_agent(self) -> AgentExecutor:
        # Try to find a search tool for self-ask pattern
        search_tool = next((tool for tool in self.tools if 'search' in tool.name.lower()), None)
        if not search_tool:
            search_tool = next((tool for tool in self.tools if 'external' in tool.name.lower()), None)

        if search_tool:
            try:
                agent = create_self_ask_with_search_agent(self.llm, [search_tool], verbose=True)
                return AgentExecutor(agent=agent, tools=[search_tool], verbose=True)
            except Exception:
                # Fallback to ReAct if self-ask fails
                pass

        # Fallback: Use ReAct with self-questioning prompt
        prompt = self.get_prompt_template()
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=settings.MAX_ITERATIONS,
            handle_parsing_errors=True
        )


class AgentFactory:
    """Factory class for creating different agent patterns"""

    _pattern_classes = {
        AgentPattern.REACT: ReactAgentPattern,
        AgentPattern.PLAN_EXECUTE: PlanExecuteAgentPattern,
        AgentPattern.MULTI_AGENT: MultiAgentPattern,
        AgentPattern.CONVERSATIONAL: ConversationalAgentPattern,
        AgentPattern.SELF_ASK: SelfAskAgentPattern,
    }

    @classmethod
    def create_agent_pattern(
            cls,
            pattern: AgentPattern,
            vector_store: VectorStoreManager,
            tools: Optional[List[BaseTool]] = None
    ) -> BaseAgentPattern:
        """Create an agent pattern instance"""

        if tools is None:
            tools = get_all_tools(vector_store)

        pattern_class = cls._pattern_classes.get(pattern)
        if not pattern_class:
            raise ValueError(f"Unsupported agent pattern: {pattern}")

        return pattern_class(vector_store, tools)

    @classmethod
    def get_available_patterns(cls) -> List[AgentPattern]:
        """Get list of available agent patterns"""
        return list(cls._pattern_classes.keys())

    @classmethod
    def get_pattern_description(cls, pattern: AgentPattern) -> str:
        """Get description of an agent pattern"""
        descriptions = {
            AgentPattern.REACT: "Reasoning + Acting: Thinks step by step and uses tools iteratively",
            AgentPattern.PLAN_EXECUTE: "Plan then Execute: Creates a plan first, then executes each step",
            AgentPattern.MULTI_AGENT: "Multiple Specialists: Uses specialized agents working together",
            AgentPattern.CONVERSATIONAL: "Conversational: Maintains context and memory across interactions",
            AgentPattern.SELF_ASK: "Self-Ask: Breaks down complex questions into sub-questions",
        }
        return descriptions.get(pattern, "Unknown pattern")