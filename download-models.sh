#!/bin/bash
test_model_dir="./test/data"

mkdir -p $test_model_dir

resnet_file="resnet_v1_50_2016_08_28.tar.gz"
if ! [ -f $resnet_file ]; then
	wget http://download.tensorflow.org/models/$resnet_file
	tar -xf $resnet_file -C $test_model_dir
fi

vgg_file="vgg_16_2016_08_28.tar.gz"
if ! [ -f $vgg_file ]; then
	wget http://download.tensorflow.org/models/$vgg_file
	tar -xf $vgg_file -C $test_model_dir
fi

text_sequence_file="sequence.tar"
if ! [ -f $text_sequence_file ]; then
	wget https://raw.githubusercontent.com/darkonhub/darkon-examples/master/gradcam/$text_sequence_file
	tar -xf $text_sequence_file -C $test_model_dir
fi

