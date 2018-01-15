test_model_dir="./test/data"

wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xf resnet_v1_50_2016_08_28.tar.gz -C $test_model_dir
tar -xf vgg_16_2016_08_28.tar.gz -C $test_model_dir

