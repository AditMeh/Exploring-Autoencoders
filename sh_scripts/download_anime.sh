mkdir ../.kaggle
mv kaggle.json ~/.kaggle
kaggle datasets download -d splcher/animefacedataset
unzip animefacedataset.zip -d data/animefacedataset
rm -r -f animefacedataset.zip
