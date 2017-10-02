package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.image.core.BGRImageCropper
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.util.Random

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

    val source = row.getSeq[Float](0).toArray
    val dimension = row.getSeq[Int](1).toArray

    require(source.length == dimension.product)
    require(dimension.length == 3)

    val width = dimension(1)
    val height = dimension(2)
    val cH = $(cropHeight)
    val cW = $(cropWidth)
    val result = BGRImageCropper(cW, cH, $(cropperMethod)).apply((source, dimension))

    Row.fromTuple(result)
  }

  override protected def outputDataType: DataType = new StructType()
    .add(StructField("_1", new ArrayType(FloatType, false)))
    .add(StructField("_2", new ArrayType(IntegerType, false)))

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == new StructType()
      .add(StructField("_1", new ArrayType(FloatType, true)))
      .add(StructField("_2", new ArrayType(IntegerType, true)))
      || inputType == new StructType()
        .add(StructField("_1", new ArrayType(FloatType, false)))
        .add(StructField("_2", new ArrayType(IntegerType, false))),
      s"Bad input type: $inputType. Requires Struct(Array[Float], Array[Int]).")
  }

}
