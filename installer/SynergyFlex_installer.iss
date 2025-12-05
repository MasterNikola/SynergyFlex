; ---------------------------
;   SynergyFlex Installer
;   Travail de Master
;   Version 1.0.0
; ---------------------------

[Setup]
AppId={{B34E6C2E-1F00-4E91-A499-ABCDEF123456}
AppName=SynergyFlex
AppVersion=1.0.0
AppPublisher=Vujic Nikola
DefaultDirName={commonpf}\SynergyFlex
DefaultGroupName=SynergyFlex

OutputDir=.
OutputBaseFilename=SynergyFlex_Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

; Icône de l’installeur
SetupIconFile=C:\TM\SynergyFlex\synergyflex.ico

; Langue
[Languages]
Name: "francais"; MessagesFile: "compiler:Languages\French.isl"

; ---------------------------------------
;      Fichiers à installer
; ---------------------------------------
[Files]
; Icône du logiciel
Source: "C:\TM\SynergyFlex\synergyflex.ico"; DestDir: "{app}"; Flags: ignoreversion

; Script Streamlit (app principale)
Source: "C:\TM\SynergyFlex\SynergyFlex_app.py"; DestDir: "{app}"; Flags: ignoreversion

; Dossiers du projet
Source: "C:\TM\SynergyFlex\core\*"; DestDir: "{app}\core"; Flags: recursesubdirs ignoreversion
Source: "C:\TM\SynergyFlex\ui\*"; DestDir: "{app}\ui"; Flags: recursesubdirs ignoreversion

; Lanceur .bat
Source: "C:\TM\SynergyFlex\SynergyFlex_launcher.bat"; DestDir: "{app}"; Flags: ignoreversion

; ---------------------------------------
;          Raccourcis
; ---------------------------------------
[Icons]
; Menu démarrer
Name: "{group}\SynergyFlex"; Filename: "{app}\SynergyFlex_launcher.bat"; IconFilename: "{app}\synergyflex.ico"

; Bureau
Name: "{commondesktop}\SynergyFlex"; Filename: "{app}\SynergyFlex_launcher.bat"; IconFilename: "{app}\synergyflex.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Créer un raccourci sur le Bureau"; GroupDescription: "Raccourcis supplémentaires:"; Flags: unchecked

; ---------------------------------------
;       Lancement après installation
; ---------------------------------------
[Run]
Filename: "{app}\SynergyFlex_launcher.bat"; Description: "Lancer SynergyFlex"; Flags: postinstall skipifsilent
