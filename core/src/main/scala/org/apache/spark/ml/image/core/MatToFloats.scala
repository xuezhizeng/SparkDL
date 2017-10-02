package org.apache.spark.ml.image.core

import org.apache.spark.ml.image.core.ImageData.{BytesImage, FloatsImage}
import org.opencv.core.{CvType, Mat}



class MatToFloats() extends ProcessStep[cMat, FloatsImage] {
  override def apply(prev: cMat): FloatsImage = {
    MatToFloats.transform(prev)
  }
}

object MatToFloats {
  def apply(): MatToFloats = new MatToFloats()

  def transform(prev: cMat): FloatsImage = {
    if (prev.`type`() != CvType.CV_32FC3) {
      prev.convertTo(prev, CvType.CV_32FC3)
    }

    val width = prev.width()
    val height = prev.height()
    val floats = new Array[Float](width * height * 3)
    prev.get(0, 0, floats)
    (floats, height, width, prev.channels())
  }
}

