@echo off
echo === SynergyFlex Launcher ===

REM ================================
REM 1) SELECTION AUTOMATIQUE D'ANACONDA
REM ================================

REM Chemin Anaconda sur PC perso
set "CONDA_PERSO=C:\Users\infor\anaconda3"

REM Chemin Anaconda sur PC bureau
set "CONDA_BUREAU=C:\Users\vujic\anaconda3"

set "CONDA_PATH="

if exist "%CONDA_PERSO%\Scripts\activate.bat" (
    echo -> Anaconda perso détecté
    set "CONDA_PATH=%CONDA_PERSO%"
) else (
    if exist "%CONDA_BUREAU%\Scripts\activate.bat" (
        echo -> Anaconda bureau détecté
        set "CONDA_PATH=%CONDA_BUREAU%"
    )
)

if "%CONDA_PATH%"=="" (
    echo ERREUR : Aucun environnement Anaconda trouvé.
    pause
    exit /b
)

call "%CONDA_PATH%\Scripts\activate.bat" base


REM ================================
REM 2) SELECTION AUTOMATIQUE DU PROJET
REM ================================

REM 1) Chemin sur lecteur M: (ce que tu utilises et qui MARCHE)
set "SYNERGY_M=M:\TM_synergyFlex_Nikola\SynergyFlex"

REM 2) Chemin reseau direct (au cas où M: n'existe pas)
set "SYNERGY_SERVER=\\ds416_ant_1\INGCVC_MANDATS\TM_SynergyFlex_Nikola\SynergyFlex"

REM 3) Copie locale possible sur PC perso
set "SYNERGY_LOCAL=C:\TM\SynergyFlex"

set "SYNERGY_PATH="

if exist "%SYNERGY_M%\SynergyFlex_app.py" (
    echo -> Projet trouvé sur M:
    set "SYNERGY_PATH=%SYNERGY_M%"
) else (
    if exist "%SYNERGY_SERVER%\SynergyFlex_app.py" (
        echo -> Projet trouvé sur le serveur
        set "SYNERGY_PATH=%SYNERGY_SERVER%"
    ) else (
        if exist "%SYNERGY_LOCAL%\SynergyFlex_app.py" (
            echo -> Projet trouvé en local
            set "SYNERGY_PATH=%SYNERGY_LOCAL%"
        )
    )
)

if "%SYNERGY_PATH%"=="" (
    echo ERREUR : Impossible de localiser SynergyFlex.
    pause
    exit /b
)


REM ================================
REM 3) LANCEMENT DE L'APPLICATION
REM ================================

cd /d "%SYNERGY_PATH%"
echo Lancement de SynergyFlex depuis : %SYNERGY_PATH%
streamlit run SynergyFlex_app.py

pause
