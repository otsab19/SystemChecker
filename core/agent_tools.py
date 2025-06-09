from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any, List
import json
from core.system_collector import SystemDataCollector
from core.vector_store import VectorStoreManager
from utils.platform_utils import PlatformUtils
from config.settings import settings

# Global instances
_vector_store = None
_llm = None
_collector = None
_platform_utils = None


def initialize_tools(vector_store: VectorStoreManager):
    """Initialize global tool dependencies"""
    global _vector_store, _llm, _collector, _platform_utils
    _vector_store = vector_store
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.TEMPERATURE
    )
    _collector = SystemDataCollector()
    _platform_utils = PlatformUtils()


@tool
def rag_query(query: str) -> str:
    """Query the local system information database for relevant information"""
    try:
        results = _vector_store.query_similar(query, n_results=5)

        if not results:
            return "No relevant system information found for your query."

        context = "\n\n".join([result["content"] for result in results])

        prompt = f"""
Based on the following system information, answer the user's query: "{query}"

System Information:
{context}

Please provide a helpful and accurate response based on the system data provided.
Include specific metrics, timestamps, and actionable recommendations when available.
"""

        response = _llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error querying system information: {str(e)}"


@tool
def live_system_info(command_or_request: str) -> str:
    """Get real-time system information or execute specific system commands"""
    try:
        request_lower = command_or_request.lower()

        if "cpu" in request_lower and "usage" in request_lower:
            return _get_cpu_usage()
        elif "memory" in request_lower or "ram" in request_lower:
            return _get_memory_info()
        elif "disk" in request_lower:
            return _get_disk_info()
        elif "process" in request_lower:
            return _get_top_processes()
        elif "network" in request_lower:
            return _get_network_status()
        elif "temperature" in request_lower or "thermal" in request_lower:
            return _get_thermal_info()
        elif "battery" in request_lower:
            return _get_battery_info()
        else:
            return _execute_safe_command(command_or_request)

    except Exception as e:
        return f"Error getting live system information: {str(e)}"


@tool
def system_action(command: str) -> str:
    """Execute system-level commands that modify the system (requires user confirmation)"""
    try:
        if not settings.REQUIRE_CONFIRMATION:
            result = _platform_utils.run_command(command)
            if result["success"]:
                return f"Command executed: {result['stdout']}"
            else:
                return f"Command failed: {result['stderr']}"

        print(f"\n⚠️  SYSTEM ACTION REQUESTED ⚠️")
        print(f"Command: {command}")
        print("This command will modify your system.")

        confirmation = input("Do you want to proceed? (yes/no): ").lower().strip()

        if confirmation not in ['yes', 'y']:
            return "System action cancelled by user."

        result = _platform_utils.run_command(command)

        if result["success"]:
            return f"Command executed successfully:\n{result['stdout']}"
        else:
            return f"Command failed:\n{result['stderr']}"

    except Exception as e:
        return f"Error executing system action: {str(e)}"


@tool
def external_search(query: str) -> str:
    """Search for IT knowledge and solutions online"""
    # Enhanced placeholder with structured response
    return f"""
External search results for: "{query}"

Recommended resources:
1. Official Documentation: Check vendor documentation for {query}
2. Community Forums: Search Stack Overflow, Reddit r/sysadmin
3. Knowledge Bases: Microsoft Docs, Red Hat Documentation, Ubuntu Wiki
4. Security Advisories: CVE databases, vendor security bulletins

Suggested search terms: {query}, troubleshooting, best practices, configuration

Note: This is a placeholder. In production, integrate with:
- Google Search API
- Stack Overflow API  
- Documentation APIs
- Security databases
"""


@tool
def system_health_check() -> str:
    """Perform comprehensive system health check"""
    try:
        health_report = []

        # CPU Health
        cpu_usage = _get_cpu_usage()
        health_report.append(f"CPU Status: {cpu_usage}")

        # Memory Health
        memory_info = _get_memory_info()
        health_report.append(f"Memory Status: {memory_info}")

        # Disk Health
        disk_info = _get_disk_info()
        health_report.append(f"Disk Status: {disk_info}")

        # Network Health
        network_status = _get_network_status()
        health_report.append(f"Network Status: {network_status}")

        # System Uptime
        uptime_info = _get_uptime_info()
        health_report.append(f"Uptime: {uptime_info}")

        return "\n\n".join(health_report)

    except Exception as e:
        return f"Error performing health check: {str(e)}"


@tool
def security_scan() -> str:
    """Perform basic security assessment"""
    try:
        security_report = []

        # Check for running security services
        security_services = _check_security_services()
        security_report.append(f"Security Services: {security_services}")

        # Check for open ports
        open_ports = _check_open_ports()
        security_report.append(f"Open Ports: {open_ports}")

        # Check for recent security logs
        security_logs = _check_security_logs()
        security_report.append(f"Recent Security Events: {security_logs}")

        return "\n\n".join(security_report)

    except Exception as e:
        return f"Error performing security scan: {str(e)}"


# Helper functions
def _get_cpu_usage() -> str:
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    avg_cpu = sum(cpu_percent) / len(cpu_percent)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    result = f"CPU Usage: {avg_cpu:.1f}% average across {cpu_count} cores\n"
    result += f"Per-core usage: {[f'{cpu:.1f}%' for cpu in cpu_percent]}\n"
    if cpu_freq:
        result += f"CPU Frequency: {cpu_freq.current:.0f} MHz (max: {cpu_freq.max:.0f} MHz)"

    return result


def _get_memory_info() -> str:
    import psutil
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    result = f"Physical Memory: {memory.used / (1024 ** 3):.2f}GB / {memory.total / (1024 ** 3):.2f}GB ({memory.percent}%)\n"
    result += f"Available: {memory.available / (1024 ** 3):.2f}GB\n"
    result += f"Swap Memory: {swap.used / (1024 ** 3):.2f}GB / {swap.total / (1024 ** 3):.2f}GB ({swap.percent}%)"

    return result


def _get_disk_info() -> str:
    import psutil
    disk_info = []

    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_info.append(
                f"{partition.device} ({partition.fstype}): "
                f"{usage.used / (1024 ** 3):.2f}GB / {usage.total / (1024 ** 3):.2f}GB "
                f"({usage.percent:.1f}% used)"
            )
        except PermissionError:
            continue

    # Add disk I/O stats
    try:
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_info.append(f"\nDisk I/O: Read {disk_io.read_bytes / (1024 ** 3):.2f}GB, "
                             f"Write {disk_io.write_bytes / (1024 ** 3):.2f}GB")
    except:
        pass

    return "\n".join(disk_info)


def _get_top_processes() -> str:
    import psutil
    processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by CPU usage
    top_cpu = sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:10]

    # Sort by memory usage
    top_memory = sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:10]

    result = "Top 10 Processes by CPU Usage:\n"
    for proc in top_cpu:
        result += f"- {proc['name']} (PID: {proc['pid']}) - CPU: {proc.get('cpu_percent', 0):.1f}%, Memory: {proc.get('memory_percent', 0):.1f}%\n"

    result += "\nTop 10 Processes by Memory Usage:\n"
    for proc in top_memory:
        result += f"- {proc['name']} (PID: {proc['pid']}) - Memory: {proc.get('memory_percent', 0):.1f}%, CPU: {proc.get('cpu_percent', 0):.1f}%\n"

    return result


def _get_network_status() -> str:
    import psutil
    import socket

    # Network interfaces
    interfaces = []
    for interface_name, interface_addresses in psutil.net_if_addrs().items():
        for address in interface_addresses:
            if address.family == socket.AF_INET:
                interfaces.append(f"{interface_name}: {address.address}")

    # Network I/O stats
    try:
        net_io = psutil.net_io_counters()
        io_stats = f"\nNetwork I/O: Sent {net_io.bytes_sent / (1024 ** 2):.2f}MB, Received {net_io.bytes_recv / (1024 ** 2):.2f}MB"
    except:
        io_stats = ""

    # Active connections count
    try:
        connections = len(psutil.net_connections())
        conn_stats = f"\nActive Connections: {connections}"
    except:
        conn_stats = ""

    return "Network Interfaces:\n" + "\n".join(interfaces) + io_stats + conn_stats


def _get_thermal_info() -> str:
    """Get thermal/temperature information"""
    import psutil
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return "Temperature sensors not available on this system"

        temp_info = []
        for name, entries in temps.items():
            for entry in entries:
                temp_info.append(f"{name} - {entry.label or 'N/A'}: {entry.current}°C")

        return "System Temperatures:\n" + "\n".join(temp_info)
    except:
        return "Temperature monitoring not supported on this platform"


def _get_battery_info() -> str:
    """Get battery information"""
    import psutil
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return "No battery detected (desktop system or battery info unavailable)"

        status = "Charging" if battery.power_plugged else "Discharging"
        return f"Battery: {battery.percent}% ({status})"
    except:
        return "Battery information not available"


def _get_uptime_info() -> str:
    """Get system uptime"""
    import psutil
    from datetime import datetime

    try:
        boot_time = psutil.boot_time()
        uptime = datetime.now() - datetime.fromtimestamp(boot_time)
        return f"System uptime: {uptime}"
    except:
        return "Uptime information not available"


def _check_security_services() -> str:
    """Check security-related services"""
    # This is a simplified implementation
    return "Security services check would be implemented here (firewall, antivirus, etc.)"


def _check_open_ports() -> str:
    """Check for open network ports"""
    import psutil
    try:
        connections = psutil.net_connections(kind='inet')
        listening_ports = [conn.laddr.port for conn in connections if conn.status == 'LISTEN']
        return f"Listening ports: {sorted(set(listening_ports))}"
    except:
        return "Unable to check open ports"


def _check_security_logs() -> str:
    """Check recent security-related log entries"""
    # This would integrate with system logs
    return "Security log analysis would be implemented here"


def _execute_safe_command(command: str) -> str:
    """Execute safe read-only commands"""
    if not settings.SAFE_MODE:
        result = _platform_utils.run_command(command)
        if result["success"]:
            return result["stdout"]
        else:
            return f"Command failed: {result['stderr']}"

    # Check against whitelist
    command_start = command.split()[0] if command.split() else ""

    if command_start not in settings.ALLOWED_COMMANDS:
        return f"Command '{command}' is not in the safe commands list: {settings.ALLOWED_COMMANDS}"

    result = _platform_utils.run_command(command)
    if result["success"]:
        return result["stdout"]
    else:
        return f"Command failed: {result['stderr']}"


def get_all_tools(vector_store: VectorStoreManager) -> List:
    """Get all available tools"""
    initialize_tools(vector_store)
    return [
        rag_query,
        live_system_info,
        system_action,
        external_search,
        system_health_check,
        security_scan
    ]