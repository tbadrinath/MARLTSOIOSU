#define MyAppName "IUTMS"
#ifndef IUTMSReleaseDir
  #define IUTMSReleaseDir AddBackslash(SourcePath) + "..\..\dist\windows\release"
#endif
#ifndef IUTMSOutputDir
  #define IUTMSOutputDir AddBackslash(SourcePath) + "..\..\dist\windows\installer"
#endif

[Setup]
AppId={{6D2784E8-E521-4D2C-AB7C-57DB6B6D62D0}
AppName={#MyAppName}
AppVersion=1.0.0
DefaultDirName={autopf}\IUTMS
DefaultGroupName=IUTMS
OutputDir={#IUTMSOutputDir}
OutputBaseFilename=IUTMS-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

[Files]
Source: "{#IUTMSReleaseDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\IUTMS Telemetry Server"; Filename: "{app}\IUTMS-Server.exe"
Name: "{group}\IUTMS Simulation Launcher"; Filename: "{app}\IUTMS-GUI.exe"
Name: "{group}\Uninstall IUTMS"; Filename: "{uninstallexe}"
