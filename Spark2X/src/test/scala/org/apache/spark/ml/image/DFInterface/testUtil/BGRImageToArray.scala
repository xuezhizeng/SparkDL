package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._


class BGRImageToArray(override val uid: String)
  extends UnaryTransformer[Row, Array[Float], BGRImageToArray] {

  def this() = this(Identifiable.randomUID("BGRImageToArray"))

  final val toRGB: BooleanParam = new BooleanParam(this, "toRGB", "toRGB")

  def getToRGB: Boolean = $(toRGB)

  def setToRGB(value: Boolean): this.type = set(toRGB, value)
  setDefault(toRGB -> true)

  override protected def createTransformFunc = { row: Row =>
    val content = row.getSeq[Float](0)
    val dimension = row.getSeq[Int](1)
    val storage = new Array[Float](content.length)

    val frameLength = dimension(1) * dimension(2)
    var j = 0
    if($(toRGB)) {
      while (j < frameLength) {
        storage(j) = content(j * 3 + 2)
        storage(j + frameLength) = content(j * 3 + 1)
        storage(j + frameLength * 2) = content(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        storage(j) = content(j * 3)
        storage(j + frameLength) = content(j * 3 + 1)
        storage(j + frameLength * 2) = content(j * 3 + 2)
        j += 1
      }
    }
    storage
  }

  override protected def outputDataType: DataType = new ArrayType(FloatType, false)

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == new StructType()
      .add(new StructField("_1", new ArrayType(FloatType, false)))
      .add(new StructField("_2", new ArrayType(IntegerType, false))),
      s"Bad input type: $inputType. Requires Struct(Array[Float], Array[Int]).")
  }

}
