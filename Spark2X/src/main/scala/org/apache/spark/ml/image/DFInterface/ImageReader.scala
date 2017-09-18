package org.apache.spark.ml.image.DFInterface

import java.io.{ByteArrayInputStream, Serializable}

import org.apache.spark.ml.image.core.ReadImageUtil
import org.apache.spark.ml.image.core.ReadImageUtil
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class ImageReader extends Serializable {

  val range = 1.0F

  def readImages(path: String,
      spark: SparkSession,
      scaleTo: Int): DataFrame = {
    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (path, stream) =>
        val bytes = stream.toArray()
        val featureAndSize = ReadImageUtil.readImageFromStream(
          new ByteArrayInputStream(bytes), scaleTo, range)
        (path, featureAndSize)
    }
    images.toDF("path", "imageData")

  }

}
