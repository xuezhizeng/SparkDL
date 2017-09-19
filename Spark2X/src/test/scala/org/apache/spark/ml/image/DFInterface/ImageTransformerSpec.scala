package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.image.core.{BGRImageCropper, BGRImageNormalizer}
import org.apache.spark.sql.SparkSession
import org.scalatest.{FlatSpec, Matchers}


class ImageTransformerSpec  extends FlatSpec with Matchers{
  "ImageTransformer" should "work properly" in {

    val st = System.nanoTime()
    val spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()

    val imageDF = new ImageReader().readImages(
      "/home/yuhao/workspace/github/hhbyyh/RealEstateImage/data/*/*.jpg", spark, 256)

    val steps = BGRImageCropper(224, 224, "center") ->
      BGRImageNormalizer(Array(0.485f, 0.456f, 0.406f), Array(0.229f, 0.224f, 0.225f))

    val imgTransfomer = new ImageTransformer(steps).setInputCol("imageData").setOutputCol("feature")
    val result = imgTransfomer.transform(imageDF)
    val first = result.first()
    result.show()


  }
}



