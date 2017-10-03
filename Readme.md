
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

1. read image from file system (Local or HDFS)

   ```scala
       val imageDF = BGRImageReader.readImagesToBytes(
         "hdfs://.../data/*/*.jpg", spark, 256)
       val fileBytes = FileUtils.readFileToByteArray("/home/.../a.jpg")
       val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes, 256)   

   ```


### Spark 1X
Support RDD-based API for Spark 1.5 or Spark 1.6

### Spark 2X
Support DataFrame-based API for Spark 2.0+

1. read image from file system (Local or HDFS)

   ```scala
       val imageDF = BGRImageReader.readImagesToBytes(
         "hdfs://.../data/*/*.jpg", spark, 256)

   ```
    root
     |-- path: string (nullable = true)
     |-- imageData: struct (nullable = true)
     |    |-- _1: binary (nullable = true)
     |    |-- _2: integer (nullable = true)
     |    |-- _3: integer (nullable = true)
     |    |-- _4: integer (nullable = true)
     
     Output is a DataFrame that contains two columns.
     DataFrame("path"[String], "imageData"[Array[Bytes], height:Int, width:Int, numChannel:Int)
     
2. use BGRImageReader as a Transformer

   ```scala
      val pathDF = Seq("home/.../a.jpg",
            "home/.../b.jpg")
         .toDF("path")
      val imageDF = new BGRImageReader().setInputCol("path").setOutputCol("imageData").transform(pathDF)

   ```

     Output is a DataFrame that contains two columns.
     DataFrame("path"[String], "imageData"[Array[Bytes], height:Int, width:Int, numChannel:Int)

3. Use ImageTransformer for Image argumentation

   ```scala
      val imageDF = BGRImageReader.readImagesToBytes("hdfs://.../data/*/*.jpg", spark, 256)

    val steps = BytesToMat() ->
        Resize(250, 250) ->
        Flip(Flip.HORIZONTAL_FLIP) ->
        Cropper(224, 224) ->
        BGRImageNormalizer(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
        BGRToRGB() ->
        MatToFloats()

    val imgTransfomer = new ImageTransformer(steps)
      .setInputCol("imageData").setOutputCol("feature")


    val resultDF = imgTransfomer.transform(imageDF)

   ```


## Modules 

### Image Processing

* Easily ingest images from local or HDFS
* Assemble Image process steps with flexibility and efficiency
   
### Audio Processing

* Easily ingest audio files from local or HDFS
* Assemble Acoustic process steps with flexibility and efficiency
* Reader, Windower, DFTSpecgram, MelFrequencyFilterBank and decoders

## Building and installation

`mvn clean package`

## Contributing & feedback

contact yuhao.yang@intel.com

*Apache®, Apache Spark, and Spark® are either registered trademarks or
trademarks of the Apache Software Foundation in the United States and/or other
countries.*