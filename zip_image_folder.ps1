$sourcePath = (Get-Location).Path + "\data"
$destinationPath = (Get-Location).Path + "\data.zip"

Compress-Archive -Path "$sourcePath\*" -DestinationPath $destinationPath
