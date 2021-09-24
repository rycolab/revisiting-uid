mkdir corpora
cd corpora
curl -o cola.zip https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola.zip
curl -L -o provo.csv https://osf.io/a32be/download
curl -L -o provo_norms.csv https://osf.io/e4a2m/download
mkdir ucl
cd ucl
curl -o ucl.zip https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip
unzip ucl.zip
cd ..
curl -L -o bnc.csv https://gu-clasp.github.io/914a288ca1e127a7f1547412d9a7e056/bnc.csv
gdown https://drive.google.com/u/0/uc?id=1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8
unzip data.zip
mv data/corpora/*brown* .
rm -r data
cd ..
bash wiki-103.sh
