
Pre-processing for Deep Learning on Apache Spark
===========================================

SparkDL provides a number of feature engineering transformers to support
deep learning applications on Apache Spark. The transformed feature can
 be sent to project like BigDL.

To support different scenarios, SparkDL supports 3 set of API:

### JVM API
Allow image processing directly with Scala/Java, without Spark dependency.
This is especially helpful for low latency requirement (model serving) or
prediction on a platform other than Spark.

### Spark 1X
Support RDD-based API for Spark 1.5 or Spark 1.6

### Spark 2X
Support DataFrame-based API for Spark 2.0+

## Modules 

### Image Processing

* Easily ingest images from local or HDFS
* Assemble Image process steps with flexibility and efficiency


#### A short example

standard pre-processing for images, read, scale, crop and normalize

   ```scala
       val imageDF = new ImageReader().readImages(
         "hdfs://.../data/*/*.jpg", spark, 256)
   
       val steps = BGRImageCropper(224, 224, "center") ->
         BGRImageNormalizer(Array(0.485f, 0.456f, 0.406f), Array(0.229f, 0.224f, 0.225f))
       
       val imgTransfomer = new ImageTransformer(steps).setInputCol("imageData").setOutputCol("feature")
       imgTransfomer.transform(imageDF).show()

   ```
   
### Audio Processing

* Easily ingest audio files from local or HDFS
* Assemble Acoustic process steps with flexibility and efficiency
* Reader, Windower, DFTSpecgram, MelFrequencyFilterBank and decoders

## Building and installation

`mvn clean package`

## Contributing & feedback

Contribution are welcome.
contact yuhao.yang@intel.com

*Apache®, Apache Spark, and Spark® are either registered trademarks or
trademarks of the Apache Software Foundation in the United States and/or other
countries.*