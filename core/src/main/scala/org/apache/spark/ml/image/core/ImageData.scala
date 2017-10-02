package org.apache.spark.ml.image.core

object ImageData {

  type BytesImage = (Array[Byte], Int, Int, Int)
  type FloatsImage = (Array[Float], Int, Int, Int)

}
