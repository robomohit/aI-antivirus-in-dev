@echo off
echo "Trojan test file"
powershell -enc "CreateRemoteThread"
echo "Process injection"
VirtualAllocEx
WriteProcessMemory
OpenProcess
echo "Trojan installed"
