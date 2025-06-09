import platform
import subprocess
import json
from typing import Dict, Any, List


class PlatformUtils:
    @staticmethod
    def get_platform() -> str:
        """Get current platform (windows, linux, darwin)"""
        return platform.system().lower()

    @staticmethod
    def run_command(command: str, shell: bool = True) -> Dict[str, Any]:
        """Execute system command safely"""
        try:
            if PlatformUtils.get_platform() == "windows":
                result = subprocess.run(
                    command,
                    shell=shell,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                result = subprocess.run(
                    command.split() if not shell else command,
                    shell=shell,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }