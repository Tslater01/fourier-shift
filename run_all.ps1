$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"
$conds = @("test", "cifake4k", "ddpm", "jpeg75", "jpeg50", "jpeg30", "blur1", "blur2")

$done = @{}
if (Test-Path "results\all_evals.csv") {
    Import-Csv "results\all_evals.csv" | ForEach-Object { $done["$($_.run)|$($_.condition)"] = $true }
}

# train then immediately evaluate, so a hard stop at any point leaves a usable all_evals.csv
function Run($name, $rest) {
    if (Test-Path "results\$name\metrics.json") {
        Write-Host "skip train $name"
    }
    else {
        # a partial run left curve.csv behind; clear it or the restart double-appends epochs
        if (Test-Path "results\$name") { Remove-Item -Recurse -Force "results\$name" }
        Write-Host "=== train $name $rest"
        & $py -u -m src.train --run $name @rest
        if ($LASTEXITCODE -ne 0) { throw "train $name failed ($LASTEXITCODE)" }
    }
    foreach ($c in $conds) {
        if ($done["$name|$c"]) { continue }
        & $py -u -m src.evaluate --ckpt "results\$name\best.pt" --condition $c
        if ($LASTEXITCODE -ne 0) { throw "eval $name/$c failed ($LASTEXITCODE)" }
        $done["$name|$c"] = $true
    }
    Write-Host "--- complete $name"
}

Run "e1_rgb_s0" @("--arch", "resnet18", "--domain", "rgb", "--seed", "0")
Run "e2_fft_s0" @("--arch", "resnet18", "--domain", "fft", "--seed", "0")
Run "a4_fft_masklf4_s0" @("--arch", "resnet18", "--domain", "fft", "--mask_lf", "4", "--seed", "0")
foreach ($s in 1, 2) {
    Run "e1_rgb_s$s" @("--arch", "resnet18", "--domain", "rgb", "--seed", "$s")
    Run "e2_fft_s$s" @("--arch", "resnet18", "--domain", "fft", "--seed", "$s")
}
Run "e3_convnext_rgb_s0" @("--arch", "convnext", "--domain", "rgb", "--seed", "0")
Run "e4_convnext_fft_s0" @("--arch", "convnext", "--domain", "fft", "--seed", "0")
Run "e5_twostream_s0" @("--arch", "twostream", "--domain", "both", "--seed", "0")
Run "a1_fft_raw_s0" @("--arch", "resnet18", "--domain", "fft", "--log", "0", "--seed", "0")
Run "a2_fft_gray_s0" @("--arch", "resnet18", "--domain", "fft", "--gray", "1", "--seed", "0")
Run "a3_twostream_rgb2_s0" @("--arch", "twostream", "--domain", "rgb2", "--seed", "0")
Write-Host "queue complete"
