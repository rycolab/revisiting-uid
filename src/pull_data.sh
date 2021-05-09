curl -o cola.zip https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola.zip
curl -L -o provo.csv https://osf.io/a32be/download
mkdir ucl
cd ucl
curl -o ucl.zip https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip
unzip ucl.zip
cd ..
mkdir dundee
cd dundee
curl -o dundee.zip https://drive.switch.ch/index.php/s/gxtEy2QCKuuwl53/download
unzip dundee.zip
cd ..
curl -o enwiki.csv https://gu-clasp.github.io/b56bd338580fbb90b14e276266633abc/enwiki.csv
bash wiki-103.sh
