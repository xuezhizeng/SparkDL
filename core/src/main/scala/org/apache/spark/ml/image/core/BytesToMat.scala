package org.apache.spark.ml.image.core

import org.apache.spark.ml.image.core.ImageData.BytesImage
import org.opencv.core.{CvType, Mat}

class BytesToMat() extends ProcessStep[BytesImage, cMat] {
  override def apply(prev: BytesImage): cMat = {
    BytesToMat.transform(prev)
  }
}

object BytesToMat {
  def apply(): BytesToMat = new BytesToMat()

  def transform(prev: BytesImage): cMat = {
    val mat = new Mat(prev._2, prev._3, CvType.CV_8UC3)
    mat.put(0, 0, prev._1)
    new cMat(mat)
  }
}


