package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.image.core.ProcessStep
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import org.apache.spark.ml.image.core.ImageData.{BytesImage, FloatsImage}

class ImageTransformer(
    override val uid: String,
    val steps: ProcessStep[BytesImage, FloatsImage]
  ) extends Transformer with HasInputCol with HasOutputCol {

  def this(steps: ProcessStep[BytesImage, FloatsImage]) =
    this(Identifiable.randomUID("ImageTransformer"), steps)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def createTransformFunc: Row => Row = { row: Row =>
    val content = row.getAs[Array[Byte]](0)
    val h = row.getInt(1)
    val w = row.getInt(2)
    val c = row.getInt(3)
    require(content.length == h * w * c)
    val (target, oh, ow, oc) = steps((content, h, w, c))
    Row.fromTuple((target, oh, ow, oc))
  }

  protected def outputDataType: DataType = new StructType()
    .add(StructField("_1", new ArrayType(FloatType, false)))
    .add(StructField("_2", IntegerType))
    .add(StructField("_3", IntegerType))
    .add(StructField("_4", IntegerType))

  protected def validateInputType(inputType: DataType): Unit = {
    val validType = new StructType()
      .add(StructField("_1", BinaryType))
      .add(StructField("_2", IntegerType))
      .add(StructField("_3", IntegerType))
      .add(StructField("_4", IntegerType))
    require(inputType.sameType(validType),
      s"Bad input type: $inputType. Requires " + inputType)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val transformUDF = udf(this.createTransformFunc, outputDataType)
    dataset.withColumn($(outputCol), transformUDF(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    validateInputType(inputType)
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), outputDataType, nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): ImageTransformer = {
    val copied = new ImageTransformer(steps)
    copyValues(copied, extra)
  }
}

object ImageTransformer {

}