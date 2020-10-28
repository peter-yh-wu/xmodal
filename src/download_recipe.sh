mkdir -p ../data
cd ../data
wget http://lherranz.org/local/datasets/yummly28k/Yummly28K.rar
unrar x Yummly28K.rar
mv images27638 images
mv metadata27638 metadata
rm Yummly28K.rar
cd ../src
