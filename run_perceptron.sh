python3 perceplearn.py perceptron-training-data/train-labeled.txt 

python3 percepclassify.py vanillamodel.txt perceptron-training-data/dev-text.txt
echo "Vanilla Model Performance:"
python3 check_perform.py

python3 percepclassify.py averagedmodel.txt perceptron-training-data/dev-text.txt
echo "Average Model Performance:"
python3 check_perform.py