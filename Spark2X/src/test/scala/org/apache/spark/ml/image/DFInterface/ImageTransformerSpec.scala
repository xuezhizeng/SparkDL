package org.apache.spark.ml.image.DFInterface

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.OpenCV
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.image.core._
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

import scala.collection.mutable


class ImageTransformerSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  var spark: SparkSession = null
  val resource = getClass.getClassLoader.getResource("image/ILSVRC2012_val_00000003.JPEG")

  override def beforeAll(): Unit = {
    OpenCV.load()
    spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()
  }

  "ImageTransformer" should "work properly" in {

    val imageDF = BGRImageReader.readImagesToBytes(Paths.get(resource.toURI).toAbsolutePath.toString, spark, 256)
    imageDF.printSchema()

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

    assert(resultDF.select("feature").collect().forall {
      case Row(Row(floats: mutable.WrappedArray[Float], h: Int, w: Int, c: Int)) =>
        floats.length == h * w * c && h == 224 && w == 224 && c == 3
    })
  }

  "ImageTransformer" should "get same result" in {
    val sp = this.spark
    import sp.implicits._
    val paths = Seq(resource.getPath)
      .toDF("path")
    val imageDF = new BGRImageReader().setInputCol("path").setOutputCol("imageData").transform(paths)

    val steps = BytesToMat() -> MatToFloats()
    val imgTransfomer = new ImageTransformer(steps).setInputCol("imageData").setOutputCol("feature")
    val h1 = imgTransfomer.transform(imageDF).select("feature").rdd.map { case r: Row =>
      val internelRow = r.getAs[Row](0)
      val content = internelRow.getAs[Seq[Float]](0)
      content
    }.first()

    val reader = new BGRImageReader2()
      .setInputCol("path")
      .setOutputCol("original")
      .setNormalize(1.0f)
    val newPipeline = new Pipeline().setStages(Array(reader))
    val newPipelineModel = newPipeline.fit(paths)
    val h2 = newPipelineModel.transform(paths).select("original").rdd.map { case r: Row =>
      val internelRow = r.getAs[Row](0)
      val content = internelRow.getAs[Seq[Float]](0)
      content
    }.first()

    val correct = h1.zip(h2).count(t => Math.abs(t._1 - t._2) <= 5)
    val total = h1.length
    assert (correct.toDouble / total > 0.99)
  }
}



