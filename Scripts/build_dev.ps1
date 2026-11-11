$ErrorActionPreference = 'Stop'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
& $msbuild 'G:\Work\Spherical-Harmonic-Exponentials-for-Efficient-Glossy-Reflections\Build\ImageBasedPBR.sln' /p:Configuration=Debug /p:Platform=x64 /m /v:m /nologo
exit $LASTEXITCODE
