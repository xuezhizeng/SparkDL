package org.apache.spark.ml.image.DFInterface

import java.io.{ByteArrayInputStream, Serializable}

import org.apache.spark.ml.image.core.ReadImageUtil
import org.apache.spark.ml.image.core.ReadImageUtil
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class ImageReader extends Serializable {

  /**
   * read image from local file system or HDFS, rescale and normalize to the specific range.
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param scaleTo specify the shorter dimension after image scaling.
   * @param range maximum value to normalize pixels to the specific range. E.g. [0, 1] or [0, 255]
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Float], Array[Int] ])
   */
  def readImages(path: String,
      spark: SparkSession,
      scaleTo: Int = 256,
      range: Float = 1.0f): DataFrame = {
    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val bytes = stream.toArray()
        val featureAndSize = ReadImageUtil.readImageFromStream(
          new ByteArrayInputStream(bytes), scaleTo, range)
        (p, featureAndSize)
      }
    images.toDF("path", "imageData")
  }
}
