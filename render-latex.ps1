param (
    [string]$latexInput
)

if (!$latexInput) {
    Write-Host "❗ Please provide LaTeX equation as argument."
    exit
}

node render-latex.js "$latexInput"
