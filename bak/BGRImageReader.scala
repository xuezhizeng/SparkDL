package org.apache.spark.image

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import javax.imageio.ImageIO

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._

class BGRImageReader(override val uid: String)
  extends UnaryTransformer[String, (Array[Float], Array[Int]), BGRImageReader] {

  def this() = this(Identifiable.randomUID("BGRImageReader"))

  final val normalize: FloatParam = new FloatParam(this, "normalize", "value to be divided by each Byte")

  def getNormalize: Float = $(normalize)

  def setNormalize(value: Float): this.type = set(normalize, value)
  setDefault(normalize -> 255f)

  final val scaleTo: IntParam = new IntParam(this, "scaleTo", "scaleTo")

  def getScaleTo: Int = $(scaleTo)

  def setScaleTo(value: Int): this.type = set(scaleTo, value)
  setDefault(scaleTo -> 256)

  private def readRawImage(path: String): BufferedImage = {

    var fis: FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val image = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      require(image != null, "Can't read file " + path + ", ImageIO.read is null")
      image
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
        System.err.println("Can't read file " + path)
        throw ex
    } finally {
      if (fis != null) {
        fis.close()
      }
    }
  }

  override protected def createTransformFunc: String => (Array[Float], Array[Int]) = (path: String) => {
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
  }

  def read() = {


  }

  override protected def outputDataType: DataType =
    new StructType()
      .add(new StructField("_1", new ArrayType(FloatType, false)))
      .add(new StructField("_2", new ArrayType(IntegerType, false)))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == DataTypes.StringType, s"Bad input type: $inputType. Requires String.")
  }

}
