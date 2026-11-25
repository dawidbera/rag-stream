# VHS tape recording demonstration of rag-stream execution
Output docs/assets/demo.gif

Set FontSize 14
Set Width 1200
Set Height 680
Set Padding 20
Set Theme "Catppuccin Mocha"

Type "cd project-name"
Enter
Sleep 1s

Type "source .venv/bin/activate"
Enter
Sleep 1s

Type "pytest -v"
Enter
Sleep 4s

Type "curl -s http://localhost:8000/health | jq ."
Enter
Sleep 2s
