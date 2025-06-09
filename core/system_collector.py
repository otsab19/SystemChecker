import psutil
import platform
import json
import socket
from datetime import datetime
from typing import Dict, Any, List
from utils.platform_utils import PlatformUtils


class SystemDataCollector:
    def __init__(self):
        self.platform_utils = PlatformUtils()
        self.platform = self.platform_utils.get_platform()

    def collect_all_data(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "basic_info": self._get_basic_info(),
            "hardware_info": self._get_hardware_info(),
            "processes": self._get_process_info(),
            "services": self._get_service_info(),
            "network": self._get_network_info(),
            "logs": self._get_system_logs(),
            "installed_software": self._get_installed_software()
        }
        return data

    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic OS information"""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "uptime": self._get_uptime()
        }

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        # CPU Info
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "cpu_usage": psutil.cpu_percent(interval=1)
        }

        # Memory Info
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used,
            "free": memory.free
        }

        # Disk Info
        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "file_system": partition.fstype,
                    "total_size": partition_usage.total,
                    "used": partition_usage.used,
                    "free": partition_usage.free,
                    "percentage": (partition_usage.used / partition_usage.total) * 100
                })
            except PermissionError:
                continue

        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "disk": disk_info
        }

    def _get_process_info(self) -> List[Dict[str, Any]]:
        """Get running processes information"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage and return top 20
        return sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:20]

    def _get_service_info(self) -> List[Dict[str, Any]]:
        """Get system services information"""
        services = []

        if self.platform == "windows":
            cmd = 'Get-Service | Select-Object Name, Status, StartType | ConvertTo-Json'
            result = self.platform_utils.run_command(f'powershell -Command "{cmd}"')
            if result["success"]:
                try:
                    services = json.loads(result["stdout"])
                except json.JSONDecodeError:
                    pass

        elif self.platform in ["linux", "darwin"]:
            # Get systemd services
            result = self.platform_utils.run_command("systemctl list-units --type=service --no-pager")
            if result["success"]:
                lines = result["stdout"].split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip() and not line.startswith('●'):
                        parts = line.split()
                        if len(parts) >= 4:
                            services.append({
                                "name": parts[0],
                                "status": parts[2],
                                "description": ' '.join(parts[4:])
                            })

        return services[:50]  # Limit to 50 services

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network information"""
        network_info = {
            "interfaces": [],
            "connections": []
        }

        # Network interfaces
        for interface_name, interface_addresses in psutil.net_if_addrs().items():
            for address in interface_addresses:
                if address.family == socket.AF_INET:
                    network_info["interfaces"].append({
                        "interface": interface_name,
                        "ip": address.address,
                        "netmask": address.netmask,
                        "broadcast": address.broadcast
                    })

        # Network connections (limit to 20)
        try:
            connections = psutil.net_connections(kind='inet')[:20]
            for conn in connections:
                network_info["connections"].append({
                    "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                    "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                    "status": conn.status,
                    "pid": conn.pid
                })
        except psutil.AccessDenied:
            pass

        return network_info

    def _get_system_logs(self) -> List[Dict[str, Any]]:
        """Get recent system logs"""
        logs = []

        if self.platform == "windows":
            # Get Windows Event Logs
            cmd = '''Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2,3} -MaxEvents 50 | 
                     Select-Object TimeCreated, Id, LevelDisplayName, Message | 
                     ConvertTo-Json'''
            result = self.platform_utils.run_command(f'powershell -Command "{cmd}"')
            if result["success"]:
                try:
                    logs = json.loads(result["stdout"])
                    if not isinstance(logs, list):
                        logs = [logs]
                except json.JSONDecodeError:
                    pass

        elif self.platform in ["linux", "darwin"]:
            # Get journalctl logs
            result = self.platform_utils.run_command(
                "journalctl --no-pager -n 50 --output=json"
            )
            if result["success"]:
                for line in result["stdout"].split('\n'):
                    if line.strip():
                        try:
                            log_entry = json.loads(line)
                            logs.append({
                                "timestamp": log_entry.get("__REALTIME_TIMESTAMP"),
                                "priority": log_entry.get("PRIORITY"),
                                "message": log_entry.get("MESSAGE"),
                                "unit": log_entry.get("_SYSTEMD_UNIT")
                            })
                        except json.JSONDecodeError:
                            continue

        return logs

    def _get_installed_software(self) -> List[Dict[str, Any]]:
        """Get installed software list"""
        software = []

        if self.platform == "windows":
            cmd = '''Get-WmiObject -Class Win32_Product | 
                     Select-Object Name, Version, Vendor | 
                     ConvertTo-Json'''
            result = self.platform_utils.run_command(f'powershell -Command "{cmd}"')
            if result["success"]:
                try:
                    software = json.loads(result["stdout"])
                    if not isinstance(software, list):
                        software = [software]
                except json.JSONDecodeError:
                    pass

        elif self.platform == "linux":
            # Try different package managers
            for cmd in ["dpkg -l", "rpm -qa", "pacman -Q"]:
                result = self.platform_utils.run_command(cmd)
                if result["success"]:
                    lines = result["stdout"].split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                software.append({
                                    "name": parts[1] if cmd.startswith("dpkg") else parts[0],
                                    "version": parts[2] if len(parts) > 2 else "unknown"
                                })
                    break

        return software[:100]  # Limit to 100 packages

    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = psutil.boot_time()
            uptime = datetime.now() - datetime.fromtimestamp(uptime_seconds)
            return str(uptime)
        except:
            return "unknown"

    def format_data_for_embedding(self, data: Dict[str, Any]) -> str:
        """Format collected data into text for embedding"""
        formatted_text = f"""
System Information Report - {data['timestamp']}

=== BASIC INFORMATION ===
Platform: {data['basic_info']['platform']}
System: {data['basic_info']['system']} {data['basic_info']['release']}
Hostname: {data['basic_info']['hostname']}
Uptime: {data['basic_info']['uptime']}

=== HARDWARE INFORMATION ===
CPU: {data['hardware_info']['cpu']['physical_cores']} physical cores, {data['hardware_info']['cpu']['total_cores']} total cores
CPU Usage: {data['hardware_info']['cpu']['cpu_usage']}%
Memory: {data['hardware_info']['memory']['used'] / (1024 ** 3):.2f}GB used / {data['hardware_info']['memory']['total'] / (1024 ** 3):.2f}GB total ({data['hardware_info']['memory']['percent']}%)

=== TOP PROCESSES ===
"""

        for proc in data['processes'][:10]:
            formatted_text += f"- {proc['name']} (PID: {proc['pid']}) - CPU: {proc.get('cpu_percent', 0)}%, Memory: {proc.get('memory_percent', 0):.1f}%\n"

        formatted_text += "\n=== RECENT SYSTEM LOGS ===\n"
        for log in data['logs'][:10]:
            if isinstance(log, dict):
                message = log.get('message', log.get('Message', 'No message'))
                formatted_text += f"- {message}\n"

        return formatted_text