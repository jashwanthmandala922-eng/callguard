@echo off
cd C:\Users\jashw\OneDrive\Documents\projects\callguard
git status
git add -A
git commit -m "CallGuard project with all paths implemented" || echo "No new changes to commit"
git remote add origin https://github.com/jashwanthmandala922-eng/callguard.git 2>nul || echo "Remote may already exist"
git branch -M main
git push -u origin main
echo "Push complete!"
pause