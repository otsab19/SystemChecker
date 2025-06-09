import os
import time
import threading
from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.layout import Layout
from rich.live import Live
import click

from core.system_collector import SystemDataCollector
from core.vector_store import VectorStoreManager
from core.agent_factory import AgentFactory, BaseAgentPattern
from core.memory_manager import MemoryManager
from core.cache_manager import CacheManager
from config.settings import settings, AgentPattern, AgentMode


class AdvancedAISystemAdmin:
    def __init__(self):
        self.console = Console()
        self.collector = SystemDataCollector()
        self.vector_store = VectorStoreManager()
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager() if settings.ENABLE_CACHING else None

        # Agent pattern management
        self.current_pattern = settings.DEFAULT_AGENT_PATTERN
        self.agent_pattern: BaseAgentPattern = None
        self.initialize_agent()

        self.last_collection_time = None
        self.running = True
        self.session_start_time = datetime.now()

        # Enhanced question categories
        self.question_categories = {
            "Performance": {
                "1": "What's my current CPU and memory usage?",
                "2": "Show me the top processes consuming resources",
                "3": "Perform a comprehensive system health check",
                "4": "What's my disk space and I/O performance?",
                "5": "How can I optimize my system performance?"
            },
            "Security": {
                "6": "Perform a basic security scan",
                "7": "Check for any security concerns or recommendations",
                "8": "What security services are running?",
                "9": "Show me recent security-related events",
                "10": "Are there any open network ports I should know about?"
            },
            "Troubleshooting": {
                "11": "Are there any recent system errors or warnings?",
                "12": "Help me diagnose a system problem",
                "13": "What services are currently running?",
                "14": "Check system logs for issues",
                "15": "Why is my system running slowly?"
            },
            "Information": {
                "16": "Show me my network configuration and status",
                "17": "What's my system uptime and basic information?",
                "18": "What software is installed on my system?",
                "19": "Show me system temperature and thermal status",
                "20": "Display battery status (if applicable)"
            }
        }

    def initialize_agent(self):
        """Initialize the agent with current pattern"""
        try:
            self.agent_pattern = AgentFactory.create_agent_pattern(
                self.current_pattern,
                self.vector_store
            )
            self.console.print(f"[green]✓ Initialized {self.current_pattern.value} agent pattern[/green]")
        except Exception as e:
            self.console.print(f"[red]Error initializing agent: {str(e)}[/red]")
            # Fallback to ReAct pattern
            self.current_pattern = AgentPattern.REACT
            self.agent_pattern = AgentFactory.create_agent_pattern(
                self.current_pattern,
                self.vector_store
            )

    def start_data_collection_thread(self):
        """Start background data collection"""
        if not settings.ENABLE_BACKGROUND_COLLECTION:
            return

        def collect_data():
            while self.running:
                try:
                    self.console.print("[dim]🔄 Collecting system data...[/dim]")

                    system_data = self.collector.collect_all_data()
                    formatted_data = self.collector.format_data_for_embedding(system_data)

                    metadata = {
                        "timestamp": system_data["timestamp"],
                        "platform": system_data["basic_info"]["system"],
                        "collection_type": "background"
                    }
                    self.vector_store.add_system_data(formatted_data, metadata)

                    self.last_collection_time = datetime.now()
                    self.console.print("[green]✓ System data collected and indexed[/green]")

                    # Clean up old cache entries
                    if self.cache_manager:
                        self.cache_manager.clear_expired()

                    time.sleep(settings.COLLECTION_INTERVAL_HOURS * 3600)

                except Exception as e:
                    self.console.print(f"[red]Error collecting data: {str(e)}[/red]")
                    time.sleep(300)

        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()

    def display_welcome(self):
        """Display enhanced welcome message"""
        welcome_text = Text()
        welcome_text.append("🤖 Advanced AI System Administrator Assistant\n", style="bold blue")
        welcome_text.append("Powered by Multiple Agent Patterns & Local Intelligence\n\n", style="dim")

        welcome_text.append(f"Current Agent Pattern: ", style="bold")
        welcome_text.append(f"{self.current_pattern.value.title()}\n", style="cyan")
        welcome_text.append(f"Description: {AgentFactory.get_pattern_description(self.current_pattern)}\n\n",
                            style="dim")

        welcome_text.append("Available Commands:\n", style="bold")
        welcome_text.append("• Type a number (1-20) to select a quick question\n")
        welcome_text.append("• Type 'questions' to see all available questions by category\n")
        welcome_text.append("• Type 'patterns' to switch agent patterns\n")
        welcome_text.append("• Type 'status' for system overview\n")
        welcome_text.append("• Type 'memory' to view conversation history\n")
        welcome_text.append("• Type 'collect' to force data collection\n")
        welcome_text.append("• Type 'cache' to manage cache\n")
        welcome_text.append("• Ask any custom system-related question\n")
        welcome_text.append("• Type 'quit' or 'exit' to leave\n")

        panel = Panel(welcome_text, title="Welcome", border_style="blue")
        self.console.print(panel)

    def display_questions_by_category(self):
        """Display questions organized by category"""
        for category, questions in self.question_categories.items():
            table = Table(title=f"📋 {category} Questions", show_header=True, header_style="bold magenta")
            table.add_column("Option", style="cyan", width=8)
            table.add_column("Question", style="green", width=60)

            for key, question in questions.items():
                table.add_row(f"[bold]{key}[/bold]", question)

            self.console.print(table)
            self.console.print()

        self.console.print("[dim]💡 Tip: Just type the number (1-20) to ask that question![/dim]")

    def display_agent_patterns(self):
        """Display available agent patterns"""
        table = Table(title="🧠 Available Agent Patterns", show_header=True, header_style="bold magenta")
        table.add_column("Pattern", style="cyan", width=20)
        table.add_column("Description", style="green", width=60)
        table.add_column("Current", style="yellow", width=10)

        for pattern in AgentFactory.get_available_patterns():
            is_current = "✓" if pattern == self.current_pattern else ""
            table.add_row(
                pattern.value.title(),
                AgentFactory.get_pattern_description(pattern),
                is_current
            )

        self.console.print(table)

    def switch_agent_pattern(self):
        """Allow user to switch agent patterns"""
        self.display_agent_patterns()

        pattern_names = [p.value for p in AgentFactory.get_available_patterns()]
        choice = Prompt.ask(
            "\n[cyan]Select agent pattern[/cyan]",
            choices=pattern_names,
            default=self.current_pattern.value
        )

        new_pattern = AgentPattern(choice)
        if new_pattern != self.current_pattern:
            self.console.print(
                f"[yellow]Switching from {self.current_pattern.value} to {new_pattern.value}...[/yellow]")
            self.current_pattern = new_pattern
            self.initialize_agent()
            self.console.print(f"[green]✓ Successfully switched to {new_pattern.value} pattern[/green]")
        else:
            self.console.print("[dim]No change made[/dim]")

    def display_status(self):
        """Display comprehensive system status"""
        layout = Layout()

        # System info
        system_info = Text()
        system_info.append(f"Platform: {self.collector.platform}\n")
        system_info.append(f"Session Duration: {datetime.now() - self.session_start_time}\n")
        system_info.append(f"Last Data Collection: {self.last_collection_time or 'Never'}\n")
        system_info.append(
            f"Background Collection: {'Enabled' if settings.ENABLE_BACKGROUND_COLLECTION else 'Disabled'}\n")

        # Agent info
        agent_info = Text()
        agent_info.append(f"Current Pattern: {self.current_pattern.value.title()}\n")
        agent_info.append(f"Agent Mode: {settings.AGENT_MODE.value.title()}\n")
        agent_info.append(f"Max Iterations: {settings.MAX_ITERATIONS}\n")
        agent_info.append(f"Safe Mode: {'Enabled' if settings.SAFE_MODE else 'Disabled'}\n")

        # Memory info
        memory_info = Text()
        session_summary = self.memory_manager.get_session_summary()
        memory_info.append(session_summary)

        # Cache info
        cache_info = Text()
        if self.cache_manager:
            cache_info.append("Cache: Enabled\n")
            cache_info.append(f"TTL: {settings.CACHE_TTL_SECONDS} seconds\n")
        else:
            cache_info.append("Cache: Disabled\n")

        layout.split_column(
            Layout(Panel(system_info, title="System Status", border_style="green")),
            Layout(Panel(agent_info, title="Agent Configuration", border_style="blue")),
            Layout(Panel(memory_info, title="Memory Status", border_style="yellow")),
            Layout(Panel(cache_info, title="Cache Status", border_style="magenta"))
        )

        self.console.print(layout)

    def display_memory_summary(self):
        """Display conversation memory"""
        session_summary = self.memory_manager.get_session_summary()
        panel = Panel(session_summary, title="💭 Conversation Memory", border_style="yellow")
        self.console.print(panel)

    def manage_cache(self):
        """Cache management interface"""
        if not self.cache_manager:
            self.console.print("[yellow]Cache is disabled[/yellow]")
            return

        action = Prompt.ask(
            "[cyan]Cache action[/cyan]",
            choices=["clear_expired", "clear_all", "status"],
            default="status"
        )

        if action == "clear_expired":
            self.cache_manager.clear_expired()
            self.console.print("[green]✓ Expired cache entries cleared[/green]")
        elif action == "clear_all":
            if Confirm.ask("[red]Clear all cache entries?[/red]"):
                self.cache_manager.clear_all()
                self.console.print("[green]✓ All cache entries cleared[/green]")
        else:
            self.console.print(f"[dim]Cache TTL: {settings.CACHE_TTL_SECONDS} seconds[/dim]")

    def force_collection(self):
        """Force immediate data collection"""
        try:
            with self.console.status("[bold green]🔄 Collecting system data..."):
                system_data = self.collector.collect_all_data()
                formatted_data = self.collector.format_data_for_embedding(system_data)

                metadata = {
                    "timestamp": system_data["timestamp"],
                    "platform": system_data["basic_info"]["system"],
                    "collection_type": "manual"
                }
                self.vector_store.add_system_data(formatted_data, metadata)

                self.last_collection_time = datetime.now()

            self.console.print("[green]✓ Data collection completed[/green]")

        except Exception as e:
            self.console.print(f"[red]Error during collection: {str(e)}[/red]")

    def handle_user_input(self, user_input: str) -> str:
        """Process user input and return the question to ask"""
        user_input = user_input.strip()

        # Check all question categories
        for category, questions in self.question_categories.items():
            if user_input in questions:
                return questions[user_input]

        # Check if it's a number
        if user_input.isdigit():
            num = int(user_input)
            if 1 <= num <= 20:
                for category, questions in self.question_categories.items():
                    if str(num) in questions:
                        return questions[str(num)]

        return user_input

    def get_contextual_suggestions(self, question: str, response: str) -> str:
        """Get contextual follow-up suggestions"""
        # Get relevant context from memory
        relevant_context = self.memory_manager.get_relevant_context(question, max_items=3)

        suggestions = []
        question_lower = question.lower()

        # Pattern-specific suggestions
        if self.current_pattern == AgentPattern.MULTI_AGENT:
            suggestions.append("💡 Multi-agent analysis provides comprehensive insights from specialists")
        elif self.current_pattern == AgentPattern.PLAN_EXECUTE:
            suggestions.append("💡 Plan-Execute pattern breaks down complex tasks systematically")

        # Content-based suggestions
        if any(keyword in question_lower for keyword in ['cpu', 'memory', 'performance']):
            suggestions.extend([
                "🔍 Try: 'Perform a comprehensive system health check'",
                "📊 Consider: Regular performance monitoring",
                "⚡ Optimization: Check startup programs and services"
            ])
        elif any(keyword in question_lower for keyword in ['error', 'problem', 'issue']):
            suggestions.extend([
                "🔍 Try: 'Check system logs for issues'",
                "🛠️ Consider: Running system diagnostics",
                "📋 Next: Document the issue for future reference"
            ])
        elif any(keyword in question_lower for keyword in ['security', 'vulnerability']):
            suggestions.extend([
                "🔒 Try: 'Perform a basic security scan'",
                "🛡️ Consider: Regular security updates",
                "📋 Review: Security best practices"
            ])

        # Memory-based suggestions
        if relevant_context:
            suggestions.append(f"📚 Related: You asked similar questions {len(relevant_context)} time(s) before")

        if not suggestions:
            suggestions = [
                "💡 Tip: Type 'questions' to see all available options",
                "🔄 Try: Different agent patterns for varied approaches",
                "📊 Consider: Regular system monitoring"
            ]

        return "\n" + "\n".join(suggestions)

    def run_cli(self):
        """Run the enhanced CLI interface"""
        self.display_welcome()

        # Ask about initial setup
        if Confirm.ask("\n[cyan]Would you like to see available questions?[/cyan]", default=True):
            self.display_questions_by_category()

        # Start background collection
        self.start_data_collection_thread()

        # Initial data collection
        if Confirm.ask("\n[cyan]Perform initial system data collection?[/cyan]", default=True):
            self.force_collection()

        while True:
            try:
                self.console.print()
                user_input = Prompt.ask(
                    f"[bold cyan]❯ Ask a question (Current: {self.current_pattern.value})[/bold cyan]"
                ).strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['quit', 'exit']:
                    self.running = False
                    self.console.print("[yellow]Goodbye! 👋[/yellow]")
                    break

                elif user_input.lower() == 'status':
                    self.display_status()
                    continue

                elif user_input.lower() in ['questions', 'q', 'help']:
                    self.display_questions_by_category()
                    continue

                elif user_input.lower() in ['patterns', 'pattern', 'p']:
                    self.switch_agent_pattern()
                    continue

                elif user_input.lower() in ['memory', 'm']:
                    self.display_memory_summary()
                    continue

                elif user_input.lower() == 'collect':
                    self.force_collection()
                    continue

                elif user_input.lower() == 'cache':
                    self.manage_cache()
                    continue

                # Process the input
                actual_question = self.handle_user_input(user_input)

                if actual_question != user_input:
                    self.console.print(f"[dim]Processing: {actual_question}[/dim]")

                # Execute query through agent
                self.console.print("[dim]🤔 AI is analyzing your request...[/dim]")

                with self.console.status(f"[bold green]🧠 {self.current_pattern.value.title()} agent thinking..."):
                    result = self.agent_pattern.execute_query(actual_question)

                response = result.get("output", "No response generated")

                # Display response
                # Display response
                response_title = f"🤖 {self.current_pattern.value.title()} Agent Response"
                response_panel = Panel(
                    response,
                    title=response_title,
                    border_style="green",
                    padding=(1, 2)
                )
                self.console.print(response_panel)

                # Store in memory
                self.memory_manager.add_interaction(actual_question, response)

                # Show contextual suggestions
                suggestions = self.get_contextual_suggestions(actual_question, response)
                if suggestions:
                    self.console.print(Text(suggestions, style="dim"))

                # Show execution details if available
                if "steps" in result:
                    if Confirm.ask("\n[dim]Show execution details?[/dim]", default=False):
                        steps_text = "\n".join([f"Step {i + 1}: {step}" for i, step in enumerate(result["steps"])])
                        steps_panel = Panel(
                            steps_text,
                            title="🔍 Execution Steps",
                            border_style="blue",
                            padding=(1, 2)
                        )
                        self.console.print(steps_panel)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Operation cancelled[/yellow]")
                self.console.print("\n[yellow]Operation cancelled. Press Ctrl+C again to exit or continue...[/yellow]")
                if Confirm.ask("Exit application?", default=False):
                    self.running = False
                    break
                continue
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
                if settings.DEBUG:
                    self.console.print_exception()
                continue

def main():
    """Main entry point"""
    try:
        admin = AdvancedAISystemAdmin()
        admin.run_cli()
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
    except Exception as e:
        console = Console()
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        if settings.DEBUG:
            console.print_exception()

if __name__ == "__main__":
    main()