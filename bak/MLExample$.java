package org.apache.spark.ml.image;

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
