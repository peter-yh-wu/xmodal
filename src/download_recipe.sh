mkdir -p ../data
cd ../data
if [ ! -f Yummly28K.rar ]; then
    wget http://lherranz.org/local/datasets/yummly28k/Yummly28K.rar
fi
unrar x Yummly28K.rar
mv images27638 images
mv metadata27638 metadata
rm Yummly28K.rar
cd ../src
