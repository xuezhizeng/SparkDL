package org.apache.spark.ml.image.DFInterface

import org.apache.spark.sql.SparkSession

/**
  * Created by yuhao on 9/22/17.
  */
object CustomClassTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()
    import spark.implicits._
    class MyObj(val i: Int)
    implicit val myObjEncoder = org.apache.spark.sql.Encoders.kryo[MyObj]
    // ...
    val d = spark.createDataset(Seq(new MyObj(1),new MyObj(2),new MyObj(3)))
    d.show()
  }

}
