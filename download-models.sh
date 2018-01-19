test_model_dir="./test/data"

wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
wget https://raw.githubusercontent.com/darkonhub/darkon-examples/master/gradcam/sequence.tar

mkdir -p $test_model_dir
tar -xf resnet_v1_50_2016_08_28.tar.gz -C $test_model_dir
tar -xf vgg_16_2016_08_28.tar.gz -C $test_model_dir
tar -xf sequence.tar -C $test_model_dir

