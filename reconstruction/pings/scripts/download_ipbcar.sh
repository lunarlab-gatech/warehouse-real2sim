echo Creating the dataset path...

mkdir -p data
cd data

mkdir -p ipb_car
cd ipb_car

echo Downloading example data subset from IPB car dataset, 801 frames ...
echo Waiting for permission, will be released soon 
# [add link here]

echo Extracting dataset...
unzip ipbcar_test_subset.zip

rm ipbcar_test_subset.zip

cd ../../..
