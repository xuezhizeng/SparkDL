package org.apache.spark.ml.image

import java.io.{BufferedInputStream, ByteArrayInputStream}
import javax.imageio.ImageIO

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.image.DFInterface.ImageReader
import org.apache.spark.sql.SparkSession


object MLExample {

  def main(args: Array[String]): Unit = {
    val st = System.nanoTime()
    val spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()

    val imageDF = new ImageReader().readImages(
      "/home/yuhao/workspace/github/hhbyyh/RealEstateImage/data/*/*.jpg", spark, 256)

    val first = imageDF.first()
    imageDF.show()



  }

}
