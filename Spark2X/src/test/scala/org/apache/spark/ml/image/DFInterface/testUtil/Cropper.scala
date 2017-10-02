package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

/**
  * Cropper
  */
class Cropper(override val uid: String)
  extends UnaryTransformer[Row, Row, Cropper] {

  def this() = this(Identifiable.randomUID("cropper"))

  final val cropperMethod: Param[String] = new Param[String](this, "cropperMethod", "cropperMethod")

  def getCropperMethod: String = $(cropperMethod)

  def setCropperMethod(value: String): this.type = set(cropperMethod, value)
  setDefault(cropperMethod -> "random")

  final val cropWidth: IntParam = new IntParam(this, "cropWidth", "cropWidth")

  def getCropWidth: Int = $(cropWidth)

  def setCropWidth(value: Int): this.type = set(cropWidth, value)
  setDefault(cropWidth -> 224)

  final val cropHeight: IntParam = new IntParam(this, "cropHeight", "cropHeight")

  def getCropHeight: Int = $(cropHeight)

  def setCropHeight(value: Int): this.type = set(cropHeight, value)
  setDefault(cropHeight -> 224)

  override protected def createTransformFunc = { row: Row =>

    val source = row.getSeq[Float](0)
    val dimension = row.getSeq[Int](1)

    require(source.length == dimension.product)
    require(dimension.length == 3)

    val width = dimension(1)
    val height = dimension(2)
    val cH = $(cropHeight)
    val cW = $(cropWidth)

    val (startH, startW) = $(cropperMethod) match {
      case "random" =>
        ((height - cH) / 2, (width - cW) / 2)
      case "center" =>
        ((height - cH) / 2, (width - cW) / 2)
    }
    val startIndex = (startW + startH * width) * 3
    val frameLength = cW * cW
    val target = new Array[Float](dimension(0) * cW * cH)
    var i = 0
    while (i < frameLength) {
      target(i * 3 + 2) =
        source(startIndex + ((i / cW) * width + (i % cW)) * 3 + 2)
      target(i * 3 + 1) =
        source(startIndex + ((i / cW) * width + (i % cW)) * 3 + 1)
      target(i * 3) =
        source(startIndex + ((i / cW) * width + (i % cW)) * 3)
      i += 1
    }
    require(target.length == cW * cH * dimension(0))
    Row.fromTuple((target, Array(dimension(0), cW, cH)))
  }

  override protected def outputDataType: DataType = new StructType()
    .add(new StructField("_1", new ArrayType(FloatType, false)))
    .add(new StructField("_2", new ArrayType(IntegerType, false)))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == new StructType()
      .add(new StructField("_1", new ArrayType(FloatType, true)))
      .add(new StructField("_2", new ArrayType(IntegerType, true)))
      || inputType == new StructType()
        .add(new StructField("_1", new ArrayType(FloatType, false)))
        .add(new StructField("_2", new ArrayType(IntegerType, false))),
      s"Bad input type: $inputType. Requires Struct(Array[Float], Array[Int]).")
  }

}
