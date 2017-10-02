package org.apache.spark.ml.image.DFInterface

import java.io.{ByteArrayInputStream, InputStream, OutputStream, Serializable}
import java.nio.ByteBuffer
import java.nio.file.Paths

import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.image.core.{ReadImageUtil, cMat}
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Encoders, Row, SparkSession}

import org.apache.spark.ml.image.core.ImageData.{BytesImage, FloatsImage}


class BGRImageReader(override val uid: String)
  extends UnaryTransformer[String, BytesImage, BGRImageReader] {

  def this() = this(Identifiable.randomUID("BGRImageReader"))

  final val smallSideSize: IntParam = new IntParam(this, "scaleTo", "scaleTo")

  def getSmallSideSize: Int = $(smallSideSize)

  def setSmallSideSize(value: Int): this.type = set(smallSideSize, value)
  setDefault(smallSideSize -> 256)

  override protected def createTransformFunc: String => BytesImage = (path: String) => {
    try {
      val src: Path = new Path(path)
      val fs = src.getFileSystem(new Configuration())
      val is = fs.open(src)
      val fileBytes = IOUtils.toByteArray(is)
      ReadImageUtil.readImageAsBytes(fileBytes, $(smallSideSize))
    }
    catch {
      case e: Exception =>
        println("ERROR: error when reading " + path)
        null
    }
  }

  override protected def outputDataType: DataType =
    new StructType()
      .add(StructField("_1", BinaryType))
      .add(StructField("_2", IntegerType))
      .add(StructField("_3", IntegerType))
      .add(StructField("_4", IntegerType))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == DataTypes.StringType, s"Bad input type: $inputType. Requires String.")
  }

}

object BGRImageReader extends Serializable {

  /**
   * read image from local file system or HDFS, resize the specific size.
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param smallSideSize the size of the smallest side after resize
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Bytes], height:Int, width:Int, numChannel:Int)
   */
  def readImagesToBytes(path: String,
      spark: SparkSession,
      smallSideSize: Int): DataFrame = {

    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val fileBytes = stream.toArray()
        val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes, smallSideSize)
        (p, (bytes, h, w, c))
      }
    images.toDF("path", "imageData")
  }

  /**
   * read image from local file system or HDFS
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Bytes], height:Int, width:Int, numChannel:Int)
   */
  def readImagesToBytes(path: String,
      spark: SparkSession): DataFrame = {

    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val fileBytes = stream.toArray()
        val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes)
        (p, (bytes, h, w, c))
      }
    images.toDF("path", "imageData")
  }

  /**
   * read image from local file system or HDFS, rescale and normalize to the specific range.
   * @param path path to read images. Local or HDFS. Wildcard character are supported.
   * @param smallSideSize specify the shorter dimension after image scaling.
   * @param divisor divide each pixel by divisor. E.g. if divisor = 255f, each pixel is in [0, 1]
   * @return a DataFrame contains two columns.
   *         DataFrame("path"[String], "imageData"[Array[Float], height:Int, width:Int, numChannel:Int])
   */
  def readImagesAsFloats(
      path: String,
      spark: SparkSession,
      smallSideSize: Int = 256,
      divisor: Float = 1.0f): DataFrame = {

    import spark.implicits._
    val images = spark.sparkContext.binaryFiles(path)
      .map { case (p, stream) =>
        val bytes = stream.toArray()
        val (floats, h, w, c) = ReadImageUtil.readImageAsFloats(bytes, smallSideSize, divisor)
        (p, (floats, h, w, c))
      }
    images.toDF("path", "imageData")
  }



//  def readImagesToMat(path: String,
//      spark: SparkSession,
//      smallSideSize: Int = 256): DataFrame = {
//    val longBarEncoder = Encoders.tuple(Encoders.sca, Encoders.kryo[aMat])
//    implicit val myObjEncoder = org.apache.spark.sql.Encoders.kryo[(String, aMat)]
//    val pairs = spark.sparkContext.binaryFiles(path)
//      .map { case (p, stream) =>
//        val fileBytes = stream.toArray()
//        val mat = ReadImageUtil.readImageAsMat(fileBytes, smallSideSize)
//        (p, mat)
//      }
//
//    spark.createDataFrame(pairs)(longBarEncoder)
//
////    spark.createDataset(pairs.map(_._2)).toDF("imageData")
////    images.toDS
//  }
}
