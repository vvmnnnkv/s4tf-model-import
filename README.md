# ImportableLayer

Quick and dirty params importer for Swift for TensorFlow models. It relies on the coincidence that reflection API (Mirror) and KeyPathIterable can recursively enumerate model properties in the same order, thus it's possible to build map of named recursive model parameters and use corresponding key paths to update them by name. This allows to easily map arbitrary set of named parameters into arbitrary model structure (e.g. set nested model properties like "residual.conv1.conv2d.filters" from the flat params dictionary).

See example of usage is provided in this [notebook](https://github.com/vvmnnnkv/s4tf-fast-style-transfer/blob/master/Demo/Fast_Style_Transfer_with_S4TF.ipynb).
