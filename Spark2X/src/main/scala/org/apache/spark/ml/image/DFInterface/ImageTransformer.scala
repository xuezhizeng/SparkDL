package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.image.core.ProcessStep
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}


class ImageTransformer(
    override val uid: String,
    val steps: ProcessStep[(Array[Float], Array[Int]), (Array[Float], Array[Int])]
  ) extends Transformer with HasInputCol with HasOutputCol {

  def this(steps: ProcessStep[(Array[Float], Array[Int]), (Array[Float], Array[Int])]) =
    this(Identifiable.randomUID("minMaxScal"), steps)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def createTransformFunc = { row: Row =>
    val content = row.getSeq[Float](0).toArray
    val dimension = row.getSeq[Int](1).toArray
    require(content.length == dimension.product)
    val (target, outDimension) = steps((content, dimension))
    Row.fromTuple((target, outDimension))
  }

  protected def outputDataType: DataType = new StructType()
    .add(StructField("_1", new ArrayType(FloatType, false)))
    .add(StructField("_2", new ArrayType(IntegerType, false)))

  protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == new StructType()
      .add(new StructField("_1", new ArrayType(FloatType, false)))
      .add(new StructField("_2", new ArrayType(IntegerType, false))),
      s"Bad input type: $inputType. Requires Struct(Array[Float], Array[Int]).")
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
