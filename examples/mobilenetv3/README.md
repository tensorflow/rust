## About a sample image

The sample image in `sample_image` was taken by dskkato, and can be used under the same license of this repository.

## About model inputs/outputs nodes

There are several ways to verify model inputs/outputs nodes. One convenient way to interact with SavedModel format is `saved_model_cli` which is installed with tensorflow 2.x.

```sh
saved_model_cli show --dir examples/mobilenetv3 --tag serve --signature_def serving_default
...
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, -1, -1, 3)
      name: serving_default_input_1:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['Predictions'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1000)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```