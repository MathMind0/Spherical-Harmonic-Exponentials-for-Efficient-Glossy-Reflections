$gltfPath = 'G:/Work/Spherical-Harmonic-Exponentials-for-Efficient-Glossy-Reflections/Data/Meshes/Sphere.gltf'
$binPath = 'G:/Work/Spherical-Harmonic-Exponentials-for-Efficient-Glossy-Reflections/Data/Meshes/Sphere_data.bin'
$j = Get-Content $gltfPath -Raw | ConvertFrom-Json
$acc = $j.accessors[0]
$bv = $j.bufferViews[[int]$acc.bufferView]
$bin = [IO.File]::ReadAllBytes($binPath)
$off = [int]$bv.byteOffset + [int]$acc.byteOffset
$min = [double]::MaxValue
$max = [double]::MinValue
for ($i = 0; $i -lt [int]$acc.count; $i++) {
    for ($c = 0; $c -lt 3; $c++) {
        $v = [BitConverter]::ToSingle($bin, $off + ($i * 12) + ($c * 4))
        if ($v -lt $min) { $min = $v }
        if ($v -gt $max) { $max = $v }
    }
}
Write-Output ("min={0} max={1} count={2}" -f $min, $max, $acc.count)
