TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $TF_INC
echo $TF_LIB

nvcc -std=c++11 -ccbin=/usr/bin/g++-4.9 -c -o deform_conv.cu.o deform_conv.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-8.0/lib64/ --expt-relaxed-constexpr -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework



