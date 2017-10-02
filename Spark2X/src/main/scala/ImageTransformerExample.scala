import java.nio.file.Paths

import org.apache.spark.ml.image.DFInterface.{BGRImageReader, ImageTransformer}
import org.apache.spark.ml.image.core._
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable

object ImageTransformerExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()
    val resource = getClass.getClassLoader.getResource("image/ILSVRC2012_val_00000003.JPEG")
    val imageDF = BGRImageReader.readImagesToBytes(Paths.get(resource.toURI).toAbsolutePath.toString, spark, 256)

    val steps = BytesToMat() ->
      BGRToRGB() ->
      Resize(250, 250) ->
      Cropper(224, 224) ->
      MatToFloats()
    val imgTransfomer = new ImageTransformer(steps).setInputCol("imageData").setOutputCol("feature")
    val resultDF = imgTransfomer.transform(imageDF)
    resultDF.show()

  }
}
