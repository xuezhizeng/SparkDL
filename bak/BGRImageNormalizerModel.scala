package org.apache.spark.ml.image.DFInterface

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}


class BGRImageNormalizer (override val uid: String)
  extends Estimator[BGRImageNormalizerModel] with HasInputCol with HasOutputCol {

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): BGRImageNormalizerModel = {
    val rdd = dataset.select($(inputCol)).rdd
      .map(r => r.getAs[Row](0))
      .map(r => r.getSeq[Float](0))
      .map(arr => Vectors.dense(arr.map(_.toDouble).toArray))
    val scaler = new feature.StandardScaler(true, true)
    val scalerModel = scaler.fit(rdd)
    copyValues(new BGRImageNormalizerModel(uid,
      scalerModel.std.toArray.map(_.toFloat),
      scalerModel.mean.toArray.map(_.toFloat)))
  }

  override def transformSchema(schema: StructType): StructType = {
    val outputType = new StructType()
      .add(new StructField("_1", new ArrayType(FloatType, false)))
      .add(new StructField("_2", new ArrayType(IntegerType, false)))
    val outputFields = schema.fields :+ StructField($(outputCol), outputType, nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): BGRImageNormalizer = defaultCopy(extra)
}


class BGRImageNormalizerModel (
    override val uid: String,
    val mean: Array[Float],
    val std: Array[Float])
  extends Model[BGRImageNormalizerModel] with HasInputCol with HasOutputCol {

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def createTransformFunc = { row: Row =>
    val content = row.getSeq[Float](0).toArray
    val dimension = row.getSeq[Int](1).toArray
    org.apache.spark.ml.image.core.BGRImageNormalizer(mean, std).apply((content, dimension))
    Row.fromTuple((content, dimension))
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

  override def copy(extra: ParamMap): BGRImageNormalizerModel = {
    val copied = new BGRImageNormalizerModel(uid, std, mean)
    copyValues(copied, extra).setParent(parent)
  }

}
