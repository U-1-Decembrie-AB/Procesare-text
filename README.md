Instructiuni de rulare

In VS Code:
Deschizi folderul proiectului
 „File → Open Folder…” si selectezi folderul proiectului
 
Creezi si activezi un mediu virtual în terminalul VS Code

Deschide terminalul integrat
	
# Creare mediu virtual
python -m venv .venv
# Activare mediu
.\.venv\Scripts\Activate.ps1

# Actualizare pip 
pip install --upgrade pip

# Instalare dependente
pip install tensorflow numpy matplotlib scikit-learn seaborn
