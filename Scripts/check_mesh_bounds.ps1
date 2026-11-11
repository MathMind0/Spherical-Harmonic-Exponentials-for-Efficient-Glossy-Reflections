$ErrorActionPreference = 'Stop'
foreach ($name in @('Sphere', 'Cube')) {
    $path = "G:/Work/Spherical-Harmonic-Exponentials-for-Efficient-Glossy-Reflections/Data/Meshes/$name.gltf"
    $json = Get-Content $path -Raw | ConvertFrom-Json
    $posIdx = [int]$json.meshes[0].primitives[0].attributes.POSITION
    $acc = $json.accessors[$posIdx]
    Write-Output ("$name min: " + ($acc.min -join ',') + "  max: " + ($acc.max -join ','))
}
