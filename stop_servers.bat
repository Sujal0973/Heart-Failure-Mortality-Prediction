@echo off
rem This attempts to stop processes that were started by the above script:
rem it looks for processes whose command line contains 'uvicorn' or 'http.server'.

echo Stopping uvicorn and static server processes (if any)...
powershell -NoProfile -Command ^
  "$procs = Get-CimInstance Win32_Process | Where-Object { ($_.CommandLine -match 'uvicorn') -or ($_.CommandLine -match 'http.server') }; ^
   if ($procs) { $procs | ForEach-Object { Write-Host 'Stopping PID:' $_.ProcessId 'Cmd:' $_.CommandLine; Stop-Process -Id $_.ProcessId -Force } } else { Write-Host 'No matching processes found.' }"

echo Done.
pause
