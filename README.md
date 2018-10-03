# First Windows Form Application to Recognize MNIST Handwritten Digits using TensorflowSharp for C#. Using CNN Classifier.
Usage: (To Retrain model)
 -  ```
    cd python/
    python3 graph.py
    ```
 - Freeze Graph for usage:
    ```sh
        freeze_graph --input_graph=tmp/model/tf_graph.pb \
        --input_checkpoint=tmp/model/weights.ckpt --input_binary=True --output_node_names=output \
        --output_graph=output.pb
    ```
Build and Run!
The Compiled .exe can be found in bin/debug/TF Test.exe
### Don't move the compiled binary as it depends on the ouput.pb file fixed for the current folder hierarchy
Download the mnist handwriting images(28x28) from [Here](https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz?raw=true), unzip it, and use it to run the model.
else feed any 28x28 handwritten image, and it should work.
