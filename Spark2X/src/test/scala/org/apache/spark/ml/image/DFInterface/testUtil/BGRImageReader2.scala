package org.apache.spark.ml.image.DFInterface

import java.nio.ByteBuffer
import java.nio.file.Paths

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.image.core.BGRImage
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._

/**
  * Created by yuhao on 9/24/17.
  */
class BGRImageReader2(override val uid: String)
  extends UnaryTransformer[String, (Array[Float], Array[Int]), BGRImageReader2] {

  def this() = this(Identifiable.randomUID("BGRImageReader"))

  final val normalize: FloatParam = new FloatParam(this, "normalize", "value to be divided by each Byte")

  def getNormalize: Float = $(normalize)

  def setNormalize(value: Float): this.type = set(normalize, value)
  setDefault(normalize -> 255f)

  final val scaleTo: IntParam = new IntParam(this, "scaleTo", "scaleTo")

  def getScaleTo: Int = $(scaleTo)

  def setScaleTo(value: Int): this.type = set(scaleTo, value)
  setDefault(scaleTo -> 256)

  override protected def createTransformFunc: String => (Array[Float], Array[Int]) = (path: String) => {
    try {
      val bytes: Array[Byte] = BGRImage.readImage(Paths.get(path), $(scaleTo))
      val buffer = ByteBuffer.wrap(bytes)
      val width = buffer.getInt
      val height = buffer.getInt
      require(bytes.length - 8 == width * height * 3)

      val data = new Array[Float](bytes.length - 8)
      var i = 0
      while (i < data.length) {
        data(i) = (bytes(i + 8) & 0xff) / $(normalize)
        i += 1
      }
      require(data.length == 3 * width * height)
      (data, Array(3, width, height))
    } catch {
      case e: Exception =>
        println("ERROR: error when reading " + path)
        null
    }

  }

  override protected def outputDataType: DataType =
    new StructType()
      .add(new StructField("_1", new ArrayType(FloatType, false)))
      .add(new StructField("_2", new ArrayType(IntegerType, false)))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == DataTypes.StringType, s"Bad input type: $inputType. Requires String.")
  }

}
