# Synthesises one WAV per turn with SAPI. Two voices so the two speakers are
# distinguishable to a listener -- though NOT to the pipeline, which separates
# speakers by DEVICE, not by voice. Everything played out of one render device
# lands on the interviewee channel. That limitation is the point of measuring
# it rather than assuming it.
param(
  [Parameter(Mandatory = $true)][string]$TurnsJson,
  [Parameter(Mandatory = $true)][string]$OutDir
)

Add-Type -AssemblyName System.Speech

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$turns = Get-Content $TurnsJson -Raw -Encoding UTF8 | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# 16 kHz, 16-bit, mono -- the pipeline's own target sample rate, so nothing
# has to be resampled on the way in.
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

$index = 0
foreach ($turn in $turns) {
  $voice = if ($turn.speaker -eq 'interviewer') { 'Microsoft David Desktop' } else { 'Microsoft Zira Desktop' }
  $synth.SelectVoice($voice)
  # Slightly under default. Default SAPI reads faster than people talk, and a
  # transcription test on unnaturally fast speech measures the wrong thing.
  $synth.Rate = -1
  $path = Join-Path $OutDir ("turn_{0:d4}.wav" -f $index)
  $synth.SetOutputToWaveFile($path, $fmt)
  $synth.Speak($turn.text)
  $synth.SetOutputToNull()
  $index++
}
$synth.Dispose()
Write-Output "synthesised $index turns into $OutDir"
