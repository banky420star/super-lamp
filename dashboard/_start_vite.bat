@echo off
cd /d "C:\Users\Administrator\Desktop\SupremeChainsaw_Clean\03_UI_Monitoring\frontend"
npx vite --port 4180 --host 0.0.0.0 > vite_output.log 2>&1
